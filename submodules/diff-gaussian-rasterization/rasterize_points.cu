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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream> // 字符串流
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

// 返回值是一个可调用对象（函数、函数对象或lambda表达式），接受一个size_t类型的参数N并返回一个char*类型的指针
std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
	// 生成一个“调整张量大小并返回其内存指针”的函数
	// 最后张量的内存大小是 N * sizeof(element) 字节
	// 用于CUDA光栅化器的内存管理

	// 捕获t的引用，并返回一个lambda函数
    auto lambda = [&t](size_t N) {
		// resize_ 需要一个 std::vector<int64_t> 或者类似的形状参数，这里用 {(long long)N} 创建了一个只有一个元素的初始化列表，表示将张量调整为一维，长度为 N。
        t.resize_({(long long)N}); // 原地调整大小
		// t.contiguous()：确保张量是连续存储的（内存布局是连续的），如果原张量不连续，会返回一个新的连续张量副本。
		// .data_ptr()：返回指向张量数据的原始指针，类型是 void*。
		// reinterpret_cast<char*>：将 void* 指针强制转换为 char* 指针，方便按字节访问内存。
		// 这行代码的意思是：返回张量底层数据的连续内存起始地址，以 char* 形式返回。
		return reinterpret_cast<char*>(t.contiguous().data_ptr()); // 返回一个指向t的连续内存的指针
    };
    return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh, // 球谐系数
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool antialiasing,
	const bool debug)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0); // 点云中点的数量
  const int H = image_height; // 图像高度
  const int W = image_width; // 图像宽度

  auto int_opts = means3D.options().dtype(torch::kInt32); // k表示const，来源于匈牙利命名法
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  // 此时 out_invdepth.numel() 为 0，不会占用实际存储空间，后续可通过拼接等方式动态扩展。
  // 创建逆深度张量
  // out_invdepth是逆深度图
  // 逆深度 = 1 / 深度
  // 例如：深度2.5米 → 逆深度0.4
  torch::Tensor out_invdepth = torch::full({0, H, W}, 0.0, float_opts); // 第一个维度为0，意味着张量没有实际数据。这种用法常用于占位符，或为后续动态拼接做准备
  float* out_invdepthptr = nullptr;

  out_invdepth = torch::full({1, H, W}, 0.0, float_opts).contiguous();
  out_invdepthptr = out_invdepth.data<float>();

  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  
  // 1. 创建一个表示CUDA设备的对象
  torch::Device device(torch::kCUDA);

  // 2. 创建一个TensorOptions对象，指定数据类型为Byte（uint8）
  torch::TensorOptions options(torch::kByte);

  // 3-5. 使用相同的options但指定device为CUDA，创建三个空张量，形状为{0}（空张量）
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));

  // 6-8. 对这三个张量分别调用自定义函数resizeFunctional，返回可动态调整大小的函数
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0) // 检查是否有球谐函数系数
	  {
		M = sh.size(1); // 获取球谐函数的维度
      }
	  
	  // 返回渲染的点云数量
	  rendered = CudaRasterizer::Rasterizer::forward(
	    geomFunc,
		binningFunc,
		imgFunc,
	    P, degree, M,
		background.contiguous().data<float>(),
		W, H,
		means3D.contiguous().data<float>(), // 由torch::Tensor转换为float*，将底层指针传入
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data<float>(), 
		opacity.contiguous().data<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().data<float>(), 
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		out_color.contiguous().data<float>(), // 输出张量
		out_invdepthptr, // 逆深度指针，输出张量
		antialiasing, // 是否开启抗锯齿
		radii.contiguous().data<int>(), // 输出半径张量
		debug);
  }
  return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer, out_invdepth);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& opacities,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_invdepth,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R, 
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool antialiasing,
	const bool debug)
{
  const int P = means3D.size(0); // 点云中点的数量
  const int H = dL_dout_color.size(1); // 图像高度
  const int W = dL_dout_color.size(2); // 图像宽度
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options()); // 3D点坐标的梯度
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options()); // 2D点坐标的梯度
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options()); // 颜色的梯度
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options()); // 椭圆的梯度
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options()); // 不透明度的梯度
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options()); // 协方差矩阵的梯度
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options()); // 球谐系数的梯度
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options()); // 尺度的梯度
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options()); // 旋转的梯度
  torch::Tensor dL_dinvdepths = torch::zeros({0, 1}, means3D.options()); 
  
  float* dL_dinvdepthsptr = nullptr; // 点级别的梯度
  float* dL_dout_invdepthptr = nullptr; // 图像级别的梯度
  // 图像级别的逆深度梯度很好算，再反传到点级别的逆深度梯度用来指导训练过程中如何调整每个高斯点的逆深度
  // 目标：使渲染的深度图更接近真实深度
  if(dL_dout_invdepth.size(0) != 0)
  {
	dL_dinvdepths = torch::zeros({P, 1}, means3D.options());
	dL_dinvdepths = dL_dinvdepths.contiguous();
	dL_dinvdepthsptr = dL_dinvdepths.data<float>();
	dL_dout_invdepthptr = dL_dout_invdepth.data<float>();
  }

  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  background.contiguous().data<float>(),
	  W, H, 
	  means3D.contiguous().data<float>(),
	  sh.contiguous().data<float>(),
	  colors.contiguous().data<float>(),
	  opacities.contiguous().data<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  cov3D_precomp.contiguous().data<float>(),
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<float>(),
	  dL_dout_invdepthptr,
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dconic.contiguous().data<float>(),  
	  dL_dopacity.contiguous().data<float>(),
	  dL_dcolors.contiguous().data<float>(),
	  dL_dinvdepthsptr,
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dcov3D.contiguous().data<float>(),
	  dL_dsh.contiguous().data<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
	  antialiasing,
	  debug);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool)); // 标记每个高斯点的可见性
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}
