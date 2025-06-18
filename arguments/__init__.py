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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        """
        初始化参数组
        :param parser: 命令行参数解析器
        :param name: 参数组名称
        :param fill_none: 是否将默认值设置为None
        """
        group = parser.add_argument_group(name)
        for key, value in vars(self).items(): # vars() 返回对象的属性字典
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    # action="store_true"表示如果命令行中提供了该选项，则默认值为True
                    # 如果命令行中没有提供该选项，则默认值为False
                    # default=value会覆盖action="store_true"的默认行为
                    # 如果命令行没有提供该参数，值为False
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    # type=t 表示将命令行参数的类型设置为t
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3 # 球谐函数的最大阶数
        self._source_path = "" # 数据集路径
        self._model_path = "" # 模型路径
        self._images = "images" # 图像文件夹名称
        self._depths = "" # 深度文件夹名称
        self._resolution = -1 # 图像分辨率（-1表示原始分辨率）
        self._white_background = False # 是否使用白色背景
        self.train_test_exp = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        # 将source_path转换为绝对路径
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False # 是否使用Python转换球谐函数    
        self.compute_cov3D_python = False # 是否使用Python计算3D协方差
        self.debug = False # 是否启用调试模式
        self.antialiasing = False # 是否启用抗锯齿
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000 # 迭代次数
        self.position_lr_init = 0.00016 # 位置学习率初始值
        self.position_lr_final = 0.0000016 # 位置学习率最终值
        self.position_lr_delay_mult = 0.01 # 位置学习率延迟倍数
        self.position_lr_max_steps = 30_000 # 位置学习率最大步数
        self.feature_lr = 0.0025 # 特征学习率
        self.opacity_lr = 0.025 # 不透明度学习率
        self.scaling_lr = 0.005 # 缩放学习率
        self.rotation_lr = 0.001 # 旋转学习率
        self.exposure_lr_init = 0.01 # 曝光学习率初始值
        self.exposure_lr_final = 0.001 # 曝光学习率最终值
        self.exposure_lr_delay_steps = 0 # 曝光学习率延迟步数
        self.exposure_lr_delay_mult = 0.0 # 曝光学习率延迟倍数
        self.percent_dense = 0.01 # 致密化百分比
        self.lambda_dssim = 0.2 # DSSIM损失权重
        self.densification_interval = 100 # 致密化间隔
        self.opacity_reset_interval = 3000 # 不透明度重置间隔
        self.densify_from_iter = 500 # 开始致密化迭代次数
        self.densify_until_iter = 15_000 # 致密化结束迭代次数
        self.densify_grad_threshold = 0.0002 # 致密化梯度阈值
        self.depth_l1_weight_init = 1.0 # 深度L1损失权重初始值
        self.depth_l1_weight_final = 0.01 # 深度L1损失权重最终值    
        self.random_background = False # 是否使用随机背景
        self.optimizer_type = "default" # 优化器类型
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:] # 命令行参数列表
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try: # 尝试从模型路径中读取配置文件
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
