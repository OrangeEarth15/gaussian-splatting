# Differential Gaussian Rasterization

Used as the rasterization engine for the paper "3D Gaussian Splatting for Real-Time Rendering of Radiance Fields". If you can make use of it in your own research, please be so kind to cite us.

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}</code></pre>
  </div>
</section>

# diff-gaussian-rasterization的完整构建链路
项目结构概览
```text
diff-gaussian-rasterization/
├── setup.py                    # Python构建脚本
├── ext.cpp                     # C++扩展入口
├── rasterize_points.cu         # 点光栅化CUDA实现
├── rasterize_points.h          # 点光栅化头文件
├── cuda_rasterizer/            # CUDA光栅化器核心
│   ├── forward.cu             # 前向传播实现
│   ├── backward.cu            # 反向传播实现
│   ├── rasterizer_impl.cu     # 光栅化器实现
│   └── *.h                    # 头文件
├── diff_gaussian_rasterization/ # Python包
│   └── __init__.py            # 包入口
├── third_party/               # 第三方依赖
│   └── glm/                   # GLM数学库
└── CMakeLists.txt             # CMake构建配置
```
## 构建方式

```
setup.py (Python包构建)
    ↓ 调用
ext.cpp (Python-C++接口)
    ↓ 调用
CUDA源文件 (.cu)
```

### setup.py - 构建入口
```python
CUDAExtension(
    name="diff_gaussian_rasterization._C",
    sources=[
        "ext.cpp",                    # ← Python接口
        "cuda_rasterizer/forward.cu", # ← CUDA实现
        "cuda_rasterizer/backward.cu", # ← CUDA实现
        # ... 其他CUDA文件
    ]
)
```
**作用**：定义如何构建Python可导入的CUDA扩展

### ext.cpp - 接口层
```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
    m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
}
```
**作用**：将CUDA函数暴露给Python

### CMakeLists.txt - 独立构建
```cmake
add_library(CudaRasterizer
    cuda_rasterizer/forward.cu
    cuda_rasterizer/backward.cu
    # ... 其他CUDA文件
)
```
**作用**：构建独立的CUDA库（可选）

## 构建流程

### 主要流程（setup.py）
```bash
python setup.py build_ext
```
**步骤**：
1. 编译所有CUDA源文件
2. 编译ext.cpp
3. 链接生成`_C.so`
4. 安装为Python包

### 备选流程（CMake）
```bash
cmake -B build -S .
cmake --build build
```
**步骤**：
1. 构建独立的CUDA库
2. 需要手动链接到Python扩展

## 实际使用

### 在3D高斯溅射中
```python
# 使用setup.py构建
python setup.py install

# 在Python中使用
from diff_gaussian_rasterization import _C
result = _C.rasterize_gaussians(...)
```

### 4.2 文件依赖关系
```
setup.py
├── ext.cpp (Python接口)
│   ├── forward.cu (CUDA实现)
│   ├── backward.cu (CUDA实现)
│   └── rasterizer_impl.cu (CUDA实现)
└── CMakeLists.txt (可选，独立构建)
```

## 关键点

1. **setup.py是主要构建方式**：直接生成Python扩展
2. **ext.cpp是必需的**：提供Python-C++接口
3. **CMakeLists.txt是可选的**：用于独立构建CUDA库
4. **共享相同的CUDA源文件**：两种构建方式使用相同的实现

## 总结

- **setup.py**：构建Python包，包含ext.cpp和所有CUDA文件
- **ext.cpp**：Python接口，调用CUDA函数
- **CMakeLists.txt**：独立构建CUDA库（备选方案）

在3D高斯溅射项目中，主要使用setup.py方式，因为它提供了完整的Python集成。
