# PyTorch和PyTorch Geometric GPU版本安装指南

## 概述

本文档记录了在Windows环境下为VisNet-RT项目安装支持CUDA的PyTorch和PyTorch Geometric (PyG)的完整过程。该过程确保了图神经网络模型能够充分利用GPU加速训练和推理。

## 环境信息

- 操作系统: Windows 10/11
- CUDA版本: 11.8
- Python版本: 3.9.23
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU

## 安装步骤

### 1. 验证系统CUDA环境

首先检查系统中安装的CUDA驱动版本:

```bash
nvidia-smi
```

确认输出中显示的CUDA版本与要安装的PyTorch版本兼容。

### 2. 安装支持CUDA的PyTorch

卸载可能存在的CPU版本PyTorch:

```bash
pip uninstall torch torchvision torchaudio -y
```

安装支持CUDA 11.8的PyTorch版本:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

验证PyTorch的CUDA支持:

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

预期输出应显示:
- PyTorch版本: 2.7.1+cu118
- CUDA available: True
- CUDA version: 11.8

### 3. 安装PyTorch Geometric依赖项

卸载可能存在的CPU版本依赖包:

```bash
pip uninstall torch_scatter torch_sparse torch_cluster -y
```

安装支持CUDA 11.8的PyG依赖包:

```bash
pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.7.0+cu118.html
```

安装torch_spline_conv (可选):

```bash
pip install torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cu118.html
```

### 4. 安装PyTorch Geometric主包

卸载并重新安装PyTorch Geometric主包以确保兼容性:

```bash
pip uninstall torch-geometric -y
pip install torch-geometric
```

### 5. 验证安装

验证PyTorch和PyG版本:

```bash
python -c "import torch; import torch_geometric; print(f'PyTorch version: {torch.__version__}'); print(f'PyG version: {torch_geometric.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

验证PyG的CUDA支持:

```bash
python -c "from torch_geometric.data import Data; import torch; x = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float); edge_index = torch.tensor([[0, 1], [1, 2], [2, 0]], dtype=torch.long); data = Data(x=x, edge_index=edge_index.t().contiguous()); print(f'Data object created: {data}'); print(f'Data is on CUDA: {data.x.is_cuda}'); data = data.cuda(); print(f'Data after moving to CUDA: {data}'); print(f'Data is on CUDA: {data.x.is_cuda}')"
```

预期输出应显示数据对象能够成功移动到CUDA设备上。

## 常见问题及解决方案

### 1. 版本不兼容问题

确保PyTorch、PyG及其依赖项版本相互兼容。当PyTorch版本更新时，需要相应更新PyG依赖项。

### 2. CPU版本冲突

在安装GPU版本之前，务必卸载所有CPU版本的包，避免冲突。

### 3. 安装源问题

使用官方提供的预编译包URL确保获得正确的CUDA版本支持:
- PyTorch: https://download.pytorch.org/whl/
- PyG: https://data.pyg.org/whl/

## 总结

按照以上步骤，可以成功在Windows环境下为VisNet-RT项目配置支持GPU加速的PyTorch和PyTorch Geometric环境。验证过程确认了CUDA支持正常工作，可以进行后续的图神经网络模型训练和推理。