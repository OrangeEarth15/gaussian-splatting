#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

cmake_minimum_required(VERSION 3.20)

# 设置项目名称和语言
project(DiffRast LANGUAGES CUDA CXX)

# 设置C++标准和扩展
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF) # 禁用编译器特定的拓展，确保代码的可移植性

# 设置CUDA标准
set(CMAKE_CUDA_STANDARD 17)

# 保留现有的C++编译标志
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# add_library(CudaRasterizer, ...): 创建一个名为CudaRasterizer的库
# 指定库的源文件
add_library(CudaRasterizer
	cuda_rasterizer/backward.h
	cuda_rasterizer/backward.cu
	cuda_rasterizer/forward.h
	cuda_rasterizer/forward.cu
	cuda_rasterizer/auxiliary.h
	cuda_rasterizer/rasterizer_impl.cu
	cuda_rasterizer/rasterizer_impl.h
	cuda_rasterizer/rasterizer.h
)

# 设置CUDA目标架构
set_target_properties(CudaRasterizer PROPERTIES CUDA_ARCHITECTURES "70;75;86")

# 指定库的公共头文件目录
# PUBLIC表示：
# 1. 库本身可以使用这个包含目录
# 2. 链接这个库的其他目标也可以使用这个包含目录	
target_include_directories(CudaRasterizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cuda_rasterizer)

# PRIVATE表示：
# 1. 库本身可以使用这个包含目录
# 2. 链接这个库的其他目标不可以使用这个包含目录
# 这些是内部依赖，不应该暴露给外部
# 外部项目不需要知道我们使用了GLM或CUDA的具体实现
target_include_directories(CudaRasterizer PRIVATE third_party/glm ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})


