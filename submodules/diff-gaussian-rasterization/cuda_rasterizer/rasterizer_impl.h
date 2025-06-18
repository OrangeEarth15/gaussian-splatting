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

#pragma once

#include <iostream>
#include <vector>
#include "rasterizer.h"
#include <cuda_runtime_api.h>

namespace CudaRasterizer
{
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	// char*& chunk：传入一个指向内存块的指针引用，表示当前可用内存起始地址，函数内部会修改它。
	// T*& ptr：传出参数，函数会将分配的内存地址赋给它，类型为 T*。
	// std::size_t count：需要分配的元素个数。
	// std::size_t alignment：分配内存时的对齐要求（字节数），通常是2的幂。
	{
		// 将chunk的起始位置对齐到alignment的倍数，类似于num_blocks = (num_points + block_size - 1) / block_size
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count); // 更新chunk的指针位置，指向分配内存的末尾，表示剩余可用内存从这里开始
	}

	// 几何状态，存储高斯点的几何信息和投影结果
    // 处理3D到2D的投影变换
    // 计算每个高斯点的渲染参数
	struct GeometryState
	{
		size_t scan_size; // 扫描操作需要的内存大小
		float* depths; // 每个高斯点的深度值
		char* scanning_space; // 扫描算法的临时空间
		bool* clamped; // 标记点是否被视锥体裁剪
		int* internal_radii; // 计算出的内部半径
		float2* means2D; // 投射到2D屏幕的坐标
		float* cov3D; // 3D协方差矩阵（用于椭球形状）
		float4* conic_opacity; // 圆锥投影的不透明度
		float* rgb; // 每个点的RGB颜色值
		uint32_t* point_offsets; // 点在排序后的偏移
		uint32_t* tiles_touched; // 每个点接触的tile数量

		// 静态工厂方法，一种创建对象的设计模式，指的是在一个类中定义的静态方法，用来根据传入的参数或条件返回该类或相关类的实例，
		// 而不是通过new关键字来创建对象。
		static GeometryState fromChunk(char*& chunk, size_t P);
	};

    // 图像状态，管理图像级别的渲染信息
    // 处理像素级别的累积和混合
    // 管理瓦片级别的渲染范围
	struct ImageState
	{
		uint2* ranges; // 每个tile在图像中的范围（xy坐标）
		uint32_t* n_contrib; // 每个像素有多少个高斯点贡献
		float* accum_alpha; // 每个像素的累积透明度	

		static ImageState fromChunk(char*& chunk, size_t N);
	};

	// 排序状态，管理高斯点的排序和分箱
    // 优化渲染性能
    // 实现空间局部性
	struct BinningState
	{
		size_t sorting_size; // 排序操作需要的内存大小
		uint64_t* point_list_keys_unsorted; // 未排序的键
		uint64_t* point_list_keys; // 排序后的键
		uint32_t* point_list_unsorted; // 未排序的点索引
		uint32_t* point_list; // 排序后的点索引
		char* list_sorting_space; // 排序算法的临时空间

		static BinningState fromChunk(char*& chunk, size_t P);
	};

	template<typename T> 
	size_t required(size_t P)
	{
		char* size = nullptr;
		T::fromChunk(size, P); // 调用T::fromChunk(size, P)来模拟分配
		return ((size_t)size) + 128; // 返回分配的内存大小，并加上128字节作为额外空间
	}
};