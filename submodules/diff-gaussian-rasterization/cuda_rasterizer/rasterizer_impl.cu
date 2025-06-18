/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh> // 基数排序
#define GLM_FORCE_CUDA
#include <glm/glm.hpp> // glm数学库

#include <cooperative_groups.h> // CUDA协作组
#include <cooperative_groups/reduce.h> // 协作组归约
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
// 计算无符号整数n的下一个更高的最高有效位（MSB，most significant bit）的位置
// 最高有效位（MSB）：一个二进制数中最高位的1
// 例如：n = 42 (二进制: 101010)
// 位数:  5 4 3 2 1 0
// 值:    1 0 1 0 1 0
// 最高位是第5位，所以返回5
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4; // 初始猜测值，从中间开始搜
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb) // 如果n右移msb位后不为0
			msb += step; // 说明最高位在更高位置
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	// __device__端代码，判断是否在视锥体内
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
// 为每个高斯点与tile的重叠生成一个key/value对
// 每个高斯点可能影响多个tile，所以是1:N映射
// 高斯溅射渲染需要：
// 1. 确定每个高斯点影响哪些瓦片
// 2. 按瓦片和深度排序
// 3. 优化渲染性能
__global__ void duplicateWithKeys(
	int P, // 高斯点数量
	const float2* points_xy, // 每个高斯点的2D坐标（x,y）
	const float* depths, // 每个高斯点的深度值
	const uint32_t* offsets, // 每个高斯点在输出缓冲区中的偏移量
	uint64_t* gaussian_keys_unsorted, // 未排序的高斯键
	uint32_t* gaussian_values_unsorted, // 未排序的高斯值（高斯点ID）
	int* radii, // 每个高斯点的半径
	dim3 grid) // tile网格大小（grid.x, grid.y）
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		// 使用前缀和数组寻找当前高斯点在输出缓冲区中的起始位置
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		// 计算高斯点影响的tile范围
		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// 高斯点影响的瓦片范围：
		// x方向：从rect_min.x到rect_max.x
		// y方向：从rect_min.y到rect_max.y
		// 瓦片数量 = 宽度 × 高度
		// int num_tiles = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);

		// offset[idx]：第idx个高斯点结束位置
		// offset[idx-1]：第idx-1个高斯点结束位置
		// 差值：第idx个高斯点占用的位置数量
		// int positions_used = offset[idx] - offset[idx - 1];

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x; // 计算瓦片ID
				key <<= 32; // 左移32位，腾出32位给深度	
				key |= *((uint32_t*)&depths[idx]); // 将深度值转换为32位整数并合并到key中
				gaussian_keys_unsorted[off] = key; // 存储未排序的高斯键 （tile ID | depth）
				gaussian_values_unsorted[off] = idx; // 存储未排序的高斯值（高斯点ID）
				off++; // 增加偏移量
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
// 检查排序后的键，识别每个瓦片在列表中的范围
// 为每个瓦片找到其高斯点的起始和结束位置
// 排序后的列表结构：
// [瓦片0的高斯点们] [瓦片1的高斯点们] [瓦片2的高斯点们] ...
// 需要找到每个瓦片片段的边界
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
// L：排序后的列表总长度，也就是num_rendered
// point_list_keys：排序后的键列表
// ranges：存储每个tile的高斯点起始和结束位置
{
	auto idx = cg::this_grid().thread_rank(); // 每个线程处理列表里的一个位置，线程索引对应列表中的位置索引
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx]; // key：tile ID | depth
	uint32_t currtile = key >> 32; // 提取tile的ID
	if (idx == 0)
		ranges[currtile].x = 0; // 初始化tile起始位置
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom; // 复制传回，但是只复制指针
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer, // 函数指针，用于分配内存
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* out_color,
	float* depth,
	bool antialiasing,
	int* radii,
	bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy); // 计算y方向的焦距
	const float focal_x = width / (2.0f * tan_fovx); // 计算x方向的焦距

	size_t chunk_size = required<GeometryState>(P); // 计算几何状态所需的内存大小
	char* chunkptr = geometryBuffer(chunk_size); // 分配内存
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P); // 从内存块中初始化几何状态

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);  // grid size
	dim3 block(BLOCK_X, BLOCK_Y, 1); // block size

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	// 如果通道数不是3，并且没有提供预计算的高斯颜色，则抛出异常
	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	// 对每个高斯点进行预处理（变换、边界、将SH转换为RGB），不存在依赖关系
	// 主要做的就是3D点和协方差各种坐标系转换，计算高斯点影响的tile范围，在相机坐标系下的深度，求radii，mean2D，cov，inv_cov等量
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M, 
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered,
		antialiasing
	), debug)

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	// scanning_space：临时空间，用于存储前缀和  scan_size：临时空间的大小 
	// tiles_touched：每个高斯点影响的tile数量  point_offsets：每个高斯点在输出缓冲区中的偏移量  P：数组长度（高斯点数量）	
	// 计算高斯点影响瓦片数的前缀和
	// 输入：[2, 3, 0, 2, 1]  (每个高斯点影响的瓦片数)
	// 输出：[2, 5, 5, 7, 8]  (累积的瓦片数)
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	// 从point_offsets数组中获取最后一个元素的值，并将其存储到num_rendered变量中
	// 最后一个元素的值表示所有高斯点影响的tile总数，包括重复影响的tile
	// 例如，如果point_offsets数组为[2, 5, 5, 7, 8]，则num_rendered为8	
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	// 为每个需要渲染的实例生成合适的[tile | depth]键和相应的重复高斯索引
	// 输入：
	// P：高斯点数量
	// geomState.means2D：每个高斯点的2D坐标（x,y）
	// geomState.depths：每个高斯点在相机坐标系中的深度
	// geomState.point_offsets：每个高斯点在输出缓冲区中的偏移量
	// binningState.point_list_keys_unsorted：未排序的高斯键，输出变量
	// binningState.point_list_unsorted：未排序的高斯值，输出变量
	// radii：每个高斯点的半径
	// tile_grid：瓦片网格大小
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D, // 联合radii一起确定rect_min和rect_max，方便计算tile id
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid)
	CHECK_CUDA(, debug)

	// 计算瓦片网格的MSB位数
	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	// 基数排序（Radix Sort）：
	// 1. 按位排序的稳定排序算法
	// 2. 从最低位到最高位依次排序
	// 3. 适用于整数或浮点数的排序
	// 4. 时间复杂度：O(d * (n + k))，其中d是位数，n是元素数，k是基数
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug) // 起始位：0，结束位：32 + bit

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	// Let each tile blend its range of Gaussians independently in parallel
	// 每个tile独立地混合其范围内的高斯点
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges, // 每个tile的在points_list里的起始和结束位置，uint2*
		binningState.point_list,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.conic_opacity, // 逆协方差矩阵和不透明度
		imgState.accum_alpha, // 每个像素的累积alpha值
		imgState.n_contrib, // 每个像素的贡献高斯点数量
		background,
		out_color, // 输出颜色
		geomState.depths, // 每个高斯点的深度	
		depth), debug)

	return num_rendered; // 每个高斯点影响的tile数量的总和，也就是tile id ｜ 深度 的键值对数量
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	// 基本参数
	const int P, int D, int M, int R, // P：高斯点数量，D：球谐函数阶数，M：球谐函数系数数量，R：渲染的高斯点数量

	// 输入：渲染参数
	const float* background, // 背景颜色
	const int width, int height, // 图像尺寸

	// 输入：高斯点参数
	const float* means3D, // 高斯点3D坐标
	const float* shs, // 球谐函数系数
	const float* colors_precomp, // 预计算高斯颜色
	const float* opacities, // 高斯点透明度
	const float* scales, // 缩放参数
	const float scale_modifier, // 缩放修正
	const float* rotations, // 旋转参数
	const float* cov3D_precomp, // 预计算的3D协方差矩阵

	// 输入：相机参数
	const float* viewmatrix, // 观测变换矩阵
	const float* projmatrix, // 投影矩阵（观测变换+投影变换）
	const float* campos, // 相机位置
	const float tan_fovx, float tan_fovy, // 视场角

	// 输入：前向传播结果
	const int* radii, // 高斯点半径
	char* geom_buffer, // 几何缓冲区
	char* binning_buffer, // 排序缓冲区
	char* img_buffer, // 图像缓冲区

	// 输入：损失梯度
	const float* dL_dpix, // 像素梯度
	const float* dL_invdepths, // 逆深度梯度

	// 输出：参数梯度
	float* dL_dmean2D, // 2D位置梯度
	float* dL_dconic, // 圆锥参数梯度
	float* dL_dopacity, // 透明度梯度
	float* dL_dcolor, // 颜色梯度
	float* dL_dinvdepth, // 逆深度梯度
	float* dL_dmean3D, // 3D位置梯度
	float* dL_dcov3D, // 3D协方差矩阵梯度
	float* dL_dsh, // 球谐函数系数梯度
	float* dL_dscale, // 缩放梯度
	float* dL_drot, // 旋转梯度

	// 输入：反向传播参数
	bool antialiasing, // 是否启用抗锯齿
	bool debug) // 是否启用调试
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		geomState.depths,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
		dL_invdepths,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor,
		dL_dinvdepth), debug);

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		opacities,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		dL_dinvdepth,
		dL_dopacity,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot,
		antialiasing), debug);
}
