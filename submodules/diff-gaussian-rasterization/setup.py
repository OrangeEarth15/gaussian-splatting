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

from setuptools import setup # python包构建工具
# CUDAExtension 是Pytorch的CUDA扩展构建工具
# BuildExtension 是Pytorch的构建扩展工具
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__)) #获取当前文件所在目录的路径

setup(
    name="diff_gaussian_rasterization", # 包名
    packages=['diff_gaussian_rasterization'], # 包含的python包
    ext_modules=[ # 扩展模块列表
        CUDAExtension(
            name="diff_gaussian_rasterization._C", # 模块将被编译为_C
            sources=[ # 源文件列表
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            # 额外编译参数
            # nvcc: 指定nvcc编译器
            # -I: 添加包含路径
            # 添加第三方库GLM数学库路径
            extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
        ],
    cmdclass={ # 自定义命令类
        'build_ext': BuildExtension # 使用Pytorch的扩展构建命令
    }
)
