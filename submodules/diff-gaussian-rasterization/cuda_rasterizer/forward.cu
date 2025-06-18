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

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
// 将高斯点的球谐系数转换为RGB颜色
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx]; // 高斯点位置
	glm::vec3 dir = pos - campos; // 从相机到高斯点的方向
	dir = dir / glm::length(dir); // 归一化方向向量
	
	// 获取当前高斯点的球谐函数系数指针
	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs; 
	glm::vec3 result = SH_C0 * sh[0]; // 0阶：常数项（环境光）

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f; // 这个0.5的选择是经验性的，目的是将大部分球谐函数值映射到正值范围

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	// 将RGB颜色限制为正值。如果值被限制，我们需要跟踪这个信息用于反向传播。
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f); // 逐分量操作，返回的类型是glm::vec3
}

// Forward version of 2D covariance matrix computation
// 将3D空间中的高斯分布投影到2D屏幕空间，基于EWA Splatting算法实现高质量的重采样
// focal_x, focal_y: 代表的公式中near的值（由于xy轴的差异会有针对x轴的焦距和针对y轴的焦距）
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	// 考虑到了视口的宽高比和缩放，并处理了行主序和列主序的差异
	float3 t = transformPoint4x3(mean, viewmatrix); // 计算点在视锥（相机坐标系）中的位置

	// 这段代码是在进行视锥体裁剪（Frustum Culling）和坐标限制
	// 1.0倍：严格的视野边界
	// 1.3倍：给出一些缓冲区域
	// 这样可以包含那些部分可见的高斯点
	// 避免在边界处出现突然的裁剪效果
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	// 防止点超出视锥体边界，避免数值不稳定
	t.x = min(limx, max(-limx, txtz)) * t.z; // 限制x坐标在视锥范围内
	t.y = min(limy, max(-limy, tytz)) * t.z; // 限制y坐标在视锥范围内	

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0); // 雅可比矩阵，第三行忽略 

	// glm库中矩阵的存储顺序是列主序，但矩阵乘法还是按照正常进行
	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8], // 第一列
		viewmatrix[1], viewmatrix[5], viewmatrix[9], // 第二列
		viewmatrix[2], viewmatrix[6], viewmatrix[10]); // 第三列

	glm::mat3 T = W * J; // 和RS同样的道理，有点不理解

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
// 将高斯参数转换为3D世界空间协方差矩阵
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	// 构建缩放矩阵
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x; // x轴缩放
	S[1][1] = mod * scale.y; // y轴缩放
	S[2][2] = mod * scale.z; // z轴缩放

	// Normalize quaternion to get valid rotation
	// 归一化四元数以获得有效的旋转
	glm::vec4 q = rot;// / glm::length(rot);	
	float r = q.x; // 四元数实部
	float x = q.y; // 四元数虚部i
	float y = q.z; // 四元数虚部j
	float z = q.w; // 四元数虚部k

	// Compute rotation matrix from quaternion
	// 从四元数计算旋转矩阵
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	// 计算3D世界协方差矩阵Sigma，Sigma = M^T * M
	// 我有一点小疑问，原始式子不应该是 R * S * S^T * R^T 吗？
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	// 不保存左下角的三个数
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M, // P: 高斯点数量, D: 球谐函数阶数, M: 球谐函数系数数量
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	// 相机参数
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	// 输出参数
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	bool antialiasing)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	// 初始化半径和影响的tile个数为0。如果这个值没有改变，这个高斯点不会被处理
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view; // 原始高斯点在相机坐标系中的坐标
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] }; // 世界坐标系下的原始点
	float4 p_hom = transformPoint4x4(p_orig, projmatrix); // 应用投影矩阵得到齐次坐标（其实是观测变换+投影变换），因为传入的其实是full_projmatrix
	float p_w = 1.0f / (p_hom.w + 0.0000001f); 
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w }; // 透视除法：(x/w, y/w, z/w) 得到NDC坐标

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	// 如果3D协方差矩阵已经预计算，则使用它，否则从旋转和缩放参数中计算
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6; //只保留上三角的六个数
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	// 将3D空间中的高斯分布投影到2D屏幕空间，基于EWA Splatting算法实现高质量的重采样
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// 在3D点云渲染中，锯齿主要来源于：
	// 1. 离散像素采样
	// 2. 高斯椭球的锐利边缘
	// 3. 高频细节的混叠

	// 如何抗锯齿？
	// 通过增加额外的方差来"模糊"高斯椭球
	// 类似于在图像处理中的高斯模糊
	// 但需要保持能量守恒
	constexpr float h_var = 0.3f; // 抗锯齿强度，增加额外的方差项来减少锯齿，值越大，抗锯齿效果越明显，但也会使图像更模糊；0.3是经验值，在清晰度和抗锯齿效果之间取得平衡
	// constexpr表示编译时常量，提高性能
	const float det_cov = cov.x * cov.z - cov.y * cov.y;
	// 这相当于在原始协方差矩阵上加上一个单位矩阵的倍数
	// 新的协方差矩阵：Σ_new = Σ_old + h_var * I
	cov.x += h_var;
	cov.z += h_var;
	const float det_cov_plus_h_cov = cov.x * cov.z - cov.y * cov.y;
	float h_convolution_scaling = 1.0f; // 卷积缩放因子，用于调整不透明度来保持能量守恒

	// # 能量守恒原理：
	// 原始高斯分布的总能量：∫ f_old(x) dx = 1
	// 新高斯分布的总能量：∫ f_new(x) dx = 1
	// 由于我们改变了协方差矩阵，需要调整不透明度来保持能量守恒
	// 新的不透明度：opacity_new = opacity_old * sqrt(det_old / det_new)

	// 为什么是平方根？
	// 高斯分布的概率密度函数：
	// f(x) = (1/(2π√|Σ|)) * exp(-0.5 * (x-μ)^T * Σ^(-1) * (x-μ))
	// 当协方差矩阵从 Σ_old 变为 Σ_new 时：
	// 为了保持总能量不变，需要乘以 sqrt(|Σ_old| / |Σ_new|)
	if(antialiasing)
		h_convolution_scaling = sqrt(max(0.000025f, det_cov / det_cov_plus_h_cov)); // max for numerical stability

	// Invert covariance (EWA algorithm)
	const float det = det_cov_plus_h_cov;

	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv }; // 协方差矩阵的逆

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	// 计算高斯点在屏幕空间中的范围（通过找到2D协方差矩阵的特征值）
	// 使用范围来计算屏幕空间瓦片中该高斯点重叠的矩形
	// 如果矩形覆盖0个瓦片，则退出
	// λ² - trace·λ + det = 0
	// trace = cov.x + cov.z
	// det = cov.x * cov.z - cov.y * cov.y
	// lambda = （trace ± sqrt(trace² - 4 * det)） / 2
	float mid = 0.5f * (cov.x + cov.z); // 计算协方差矩阵的迹
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det)); // 计算特征值
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det)); // 计算特征值
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2))); // 计算高斯点的半径，3sigma原则：高斯分布在3个标准差之外的概率为0.0027
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) }; // 将NDC坐标转换为像素坐标，像素坐标的范围是[0, W-1]和[0, H-1](实际是[-0.5, W-0.5]和[-0.5, H-0.5])
	uint2 rect_min, rect_max; // 定义高斯点影响的tile的左上角和右下角
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	// 如果颜色已经预计算，则使用它们，否则将球谐系数转换为RGB颜色
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z; // 高斯点在相机坐标系中的深度（z坐标）
	radii[idx] = my_radius; // 高斯点在屏幕空间中的半径
	points_xy_image[idx] = point_image; // 高斯点在屏幕空间中的位置
	// Inverse 2D covariance and opacity neatly pack into one float4
	float opacity = opacities[idx]; // 高斯点的透明度

	// 协方差矩阵的逆和透明度打包成一个float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacity * h_convolution_scaling };

	// 计算高斯点影响的tile的个数（which is 矩形面积）
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
// 主光栅化方法。每个线程块协作处理一个tile，每个线程处理一个像素。交替进行数据获取和光栅化。
// 模板参数，CHANNELS表示通道数（如颜色通道数），在编译时确定，方便生成针对不同通道数的专门代码，提高性能和灵活性。
// __launch_bounds__：告诉编译器该核函数的线程块最大线程数（这里是BLOCK_X * BLOCK_Y），用于优化寄存器分配和资源使用，提升执行效率。
// 具体作用是限制线程块大小，帮助编译器生成更高效的代码，减少寄存器溢出和资源冲突
template <uint32_t CHANNELS> // CHANNELS: 通道数
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y) 
renderCUDA(
	// 每个瓦片对应的高斯点索引范围[start, end)
	const uint2* __restrict__ ranges, // 告诉编译器指针该指针指向的内存区域不会与其它指针重叠，允许编译器做更激进的优化
	const uint32_t* __restrict__ point_list, // 每个tile对应的高斯点索引列表
	int W, int H,
	const float2* __restrict__ points_xy_image, // 每个高斯点在屏幕空间中的位置
	const float* __restrict__ features, // 每个高斯点的颜色
	const float4* __restrict__ conic_opacity, // 每个高斯点的协方差矩阵和透明度
	float* __restrict__ final_T, // 累积alpha值
	uint32_t* __restrict__ n_contrib, // 累积高斯点数量
	const float* __restrict__ bg_color, // 背景颜色
	float* __restrict__ out_color, // 输出颜色
	const float* __restrict__ depths, // 每个高斯点的深度
	float* __restrict__ invdepth) // 每个高斯点的逆深度
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X; // 当前gird的列数
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y }; // 当前tile(也就是当前block)的左上角像素坐标
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) }; // 当前tile(也就是当前block)的右下角像素坐标	
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y }; // 当前线程的像素坐标	
	uint32_t pix_id = W * pix.y + pix.x; // 当前线程的像素ID	
	float2 pixf = { (float)pix.x, (float)pix.y }; // 当前线程的像素坐标的浮点数形式

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	// 如果线程在当前tile之外，则设置done为true，表示该线程不需要处理
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	// 其实block.group_index().y * horizontal_blocks + block.group_index().x表示的是当前线程块的索引
	// 等价于gridDim.x * blockIdx.y + blockIdx.x
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE); // 一个block需要迭代处理的轮数，也就是256个线程需要迭代处理的次数
	int toDo = range.y - range.x; // 当前16✖️16的tile总共需要处理的高斯点数量

	// Allocate storage for batches of collectively fetched data.
	// 一个block内的所有线程共同加载同一批高斯数据点，在这里是一共256个
	__shared__ int collected_id[BLOCK_SIZE]; // 存储当前block的高斯点ID
	__shared__ float2 collected_xy[BLOCK_SIZE]; // 存储当前block的高斯点屏幕空间坐标
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE]; // 存储当前block的协方差矩阵和透明度

	// Initialize helper variables
	float T = 1.0f; // 透射率，初始为1.0，也就是完全不被遮挡
	uint32_t contributor = 0; // 当前贡献者计数，线程局部变量	
	uint32_t last_contributor = 0; // 最后一个贡献者，线程局部变量
	float C[CHANNELS] = { 0 }; // 累积颜色值，线程局部变量

	float expected_invdepth = 0.0f; // 期望的逆深度

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		// __syncthreads_count(done)：同步+计数
		// 等待线程块内所有线程到达这个点，统计有多少线程的done为真
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank(); // 当前线程当前round负责的高斯点ID（从0开始）
		// block一起加载这批要处理的256个高斯点对应的高斯点id和屏幕空间坐标意即逆协方差矩阵和透明度
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress]; // 当前线程当前round负责的高斯点ID	
			collected_id[block.thread_rank()] = coll_id; 
			collected_xy[block.thread_rank()] = points_xy_image[coll_id]; // 当前线程当前round负责的高斯点屏幕空间坐标
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id]; // 当前线程当前round负责的高斯点协方差矩阵的逆和透明度	
		}
		block.sync();

		// Iterate over current batch
		// 这个是针对一个线程内的循环，意思是是一个线程处理一个block内的所有高斯点对本像素的影响
		// toDo -= BLOCK_SIZE，toDo的值在每一轮动态变化
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++; // 记录当前在range中的位置

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j]; // 当前block的2D高斯点在屏幕空间中的位置
			float2 d = { xy.x - pixf.x, xy.y - pixf.y }; // 当前像素与高斯点在屏幕空间中的距离
			float4 con_o = collected_conic_opacity[j]; //协方差的逆和透明度
			// 2D高斯分布：f(x,y) = (1/(2π√|Σ|)) * exp(-0.5 * (x-μ)^T * Σ^(-1) * (x-μ))
			// power = -0.5 * (x-μ)^T * Σ^(-1) * (x-μ)
			// 其中 Σ^(-1) = [a b; b c]
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y; // 计算高斯点对当前像素的影响
			// 防御性编程，过滤掉数值异常的情况
			if (power > 0.0f) 
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power)); // alpha = 基础透明度 * 高斯衰减因子（距离越远越小）
			if (alpha < 1.0f / 255.0f) // 如果alpha小于1/255，则认为该点对当前像素没有贡献，跳过
				continue;
			float test_T = T * (1 - alpha); // 计算当前像素的透射率
			// 如果 test_T < 0.0001f，说明几乎完全不透明了，渲染基本完成
			// 有点问题？这一块的判断的意思不是加上当前点之后透射率才达标吗，为什么跳过了当前点参与计算
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T; // 当前像素的RGB值 = 当前像素的RGB值 + 当前高斯点的RGB值 * 当前高斯点的透明度 * 当前像素的透射率

			if(invdepth) // 当前像素位置的平均逆深度
			expected_invdepth += (1 / depths[collected_id[j]]) * alpha * T; // 期望的逆深度 = 期望的逆深度 + 当前高斯点的逆深度 * 当前高斯点的透明度 * 当前像素的透射率

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T; // 当前像素的透射率
		n_contrib[pix_id] = last_contributor; // 记录有多少个高斯点对当前像素有贡献
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch]; // 当前像素的RGB值 = 当前像素的RGB值 + 当前像素的透射率 * 背景色

		if (invdepth)
		invdepth[pix_id] = expected_invdepth;// 1. / (expected_depth + T * 1e3);
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* depths,
	float* depth)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		depths, 
		depth);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	bool antialiasing)
{
	// 一个点一个线程，一个block处理256个点的方式来launch kernel
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered,
		antialiasing
		);
}
