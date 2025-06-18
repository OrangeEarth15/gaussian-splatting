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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer # 高斯光栅化相关的类
from scene.gaussian_model import GaussianModel # 高斯模型类
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
    """
    渲染场景
    :param viewpoint_camera: 相机视角
    :param pc（point cloud）: 高斯模型
    :param pipe: 渲染管道参数
    :param bg_color: 背景颜色
    :param scaling_modifier: 缩放修改器
    :param separate_sh: 是否分离SH
    :param override_color: 覆盖颜色
    :param use_trained_exp: 是否使用训练的曝光
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # 创建与点云坐标形状相同的零向量，设置需要梯度，形状为(N, 3)
    # +0用于确保不与原始张量共享内存
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        # 保留中间结果的梯度
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    # tanfovx：视场角的一半的正切值
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform, # 世界坐标系到相机坐标系的变换矩阵
        projmatrix=viewpoint_camera.full_proj_transform, # 世界坐标系到齐次坐标系的变换矩阵
        sh_degree=pc.active_sh_degree, 
        campos=viewpoint_camera.camera_center, 
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing # 是否使用抗锯齿
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points # 其实形状是(N, 3)
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    # 如果提供了预计算的3D协方差，则使用它。否则，它将由光栅化器从缩放/旋转计算。
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python: # 是否使用Python计算3D协方差
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling # @property
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # 如果提供了预计算的颜色（override_color），则使用它们。否则，如果想要在Python中预计算颜色，则进行计算。
    # 否则，SH -> RGB转换将由光栅化器完成。
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python: # 是否使用Python转换SHs
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # 根据是否分离球谐函数，调用光栅化器进行渲染
    if separate_sh:
        # 返回三个值：渲染后的图像，每个高斯点在屏幕上的半径，深度图
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc, 
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        # GaussianRasterizer 类继承nn.Module，相当于调用__call__方法
        # nn.Module的__call__方法会调用forward方法
        # 因此，rasterizer(...)实际上调用了GaussianRasterizer的forward方法
        # 这是PyTorch的标准行为，所有继承自nn.Module的类都可以像函数一样被调用，调用会自动转发到forward方法。
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        # 获取当前图像对应的曝光参数
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
            # 对渲染图像应用曝光变换：
            # 1. 将渲染图像的通道维度从 (C, H, W) 转换到 (H, W, C)，方便矩阵乘法，C是通道数为3
            # 2. 用曝光矩阵的前三行前三列（3x3颜色变换矩阵）乘以图像颜色向量，实现颜色调整
            # 3. 再将结果维度转换回 (C, H, W)
            # 4. 最后加上曝光矩阵前三行第4列的偏移量（颜色偏移），该偏移量广播到所有像素
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # 将渲染图像的值限制在[0, 1]之间    
    rendered_image = rendered_image.clamp(0, 1)

    # 返回渲染结果
    out = {
        "render": rendered_image, # 渲染后的图像
        "viewspace_points": screenspace_points, # 视图空间中的点
        "visibility_filter" : (radii > 0).nonzero(), # 可见性过滤器 
        "radii": radii, # 每个高斯点在屏幕上的半径
        "depth" : depth_image # 深度图
        }
    
    return out
