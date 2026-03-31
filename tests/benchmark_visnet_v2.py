#!/usr/bin/env python3
"""
VisNetV2 模型在不同特征级别下的显存占用和运行时间基准测试脚本
"""

import torch
import time
import gc
import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.visnet_v2 import VisNetV2
from core.data_preprocessor import load_datasets
from utils.feature_utils import get_feature_config
from torch_geometric.data import Batch


def load_real_data(path='./data/MMF-3/', dataname='MMF_GNN_pos', max_data=100):
    """加载真实数据"""
    try:
        # 模拟参数对象
        class Args:
            def __init__(self):
                self.model = 'visnet_v2'
                self.feature_config = 'all'
                self.standardize_features = False
                self.debug_size = max_data
        
        args = Args()
        feature_config = get_feature_config(args.feature_config)
        
        # 加载数据集
        dataset_train, dataset_dev, dataset_test, additional_features_dim, additional_features_mean, additional_features_std = load_datasets(
            args, path, dataname, max_data)
        
        # 取一部分数据用于测试
        test_dataset = dataset_test[:min(max_data, len(dataset_test))] if dataset_test else dataset_train[:min(max_data, len(dataset_train))]
        return test_dataset
    except Exception as e:
        print(f"Warning: Failed to load real data: {e}")
        print("Using synthetic data instead")
        return None


def create_test_data_from_real(real_dataset, batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """从真实数据创建测试数据"""
    if real_dataset is None:
        return create_synthetic_data(batch_size, device)
    
    try:
        # 从真实数据集中创建一个小批次
        batch_data = real_dataset[:min(batch_size, len(real_dataset))]
        
        # 批处理图数据
        batched_graph = Batch.from_data_list([data.graph for data in batch_data])
        z = batched_graph.z.to(device)
        pos = batched_graph.pos.to(device)
        batch = batched_graph.batch.to(device)
        
        # 提取特征
        physchem_features = torch.stack([data.physchem_features for data in batch_data]).to(device) if hasattr(batch_data[0], 'physchem_features') else None
        toxicity_features = torch.stack([data.toxicity_features for data in batch_data]).to(device) if hasattr(batch_data[0], 'toxicity_features') else None
        chromato_features = torch.stack([data.chromato_features for data in batch_data]).to(device) if hasattr(batch_data[0], 'chromato_features') else None
        
        return z, pos, batch, physchem_features, toxicity_features, chromato_features
    except Exception as e:
        print(f"Warning: Failed to process real data: {e}")
        print("Using synthetic data instead")
        return create_synthetic_data(batch_size, device)


def create_synthetic_data(batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """创建合成测试数据"""
    # 图结构数据
    num_atoms = batch_size * 20  # 假设每个分子平均有20个原子
    z = torch.randint(1, 10, (num_atoms,), device=device)  # 原子序数
    pos = torch.randn(num_atoms, 3, device=device)  # 3D坐标
    batch = torch.repeat_interleave(torch.arange(batch_size, device=device), 
                                    torch.full((batch_size,), 20, dtype=torch.long, device=device))  # 批次索引
    
    # 物化特征 (4维)
    physchem_features = torch.randn(batch_size, 4, device=device)
    
    # 毒性特征 (4维)
    toxicity_features = torch.randn(batch_size, 4, device=device)
    
    # 色谱特征 (2维)
    chromato_features = torch.randn(batch_size, 2, device=device)
    
    return z, pos, batch, physchem_features, toxicity_features, chromato_features


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_memory():
    """测量当前GPU显存使用情况"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # 确保所有CUDA操作完成
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        return allocated, reserved
    return 0, 0


def benchmark_model(model, z, pos, batch, physchem_features=None, toxicity_features=None, chromato_features=None, 
                   iterations=10):
    """基准测试模型"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    # 确保模型在正确的设备上
    device = z.device
    model = model.to(device)
    model.eval()
    
    # 统计模型参数量
    param_count = count_parameters(model)
    
    # 测量模型参数显存占用
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    
    # 预热运行
    with torch.no_grad():
        model(z, pos, batch, physchem_features, toxicity_features, chromato_features)
    
    # 测量前向传播显存和时间
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            output = model(z, pos, batch, physchem_features, toxicity_features, chromato_features)
    end_time = time.perf_counter()
    
    peak_memory = 0
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated()
    
    avg_time = (end_time - start_time) / iterations
    return param_count, param_memory, peak_memory, avg_time


def main():
    """主函数"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Memory measurements will be 0.")
    
    # 加载真实数据
    print("Loading real data...")
    real_dataset = load_real_data(max_data=50)  # 限制数据量以加快测试
    
    # 创建测试数据
    print("Creating test data...")
    z, pos, batch, physchem_features, toxicity_features, chromato_features = create_test_data_from_real(real_dataset, 32, device)
    
    # 四种特征级别配置
    feature_levels = ['graph', 'graph_physchem', 'graph_physchem_toxicity', 'all']
    
    results = []
    
    for feature_level in feature_levels:
        print(f"\nTesting feature level: {feature_level}")
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # 创建模型
        model = VisNetV2(
            node_feature_dim=64,
            physchem_feature_dim=4,
            toxicity_feature_dim=4,
            chromato_feature_dim=2,
            graph_hidden_dim=512,
            physchem_hidden_dim=64,
            toxicity_hidden_dim=64,
            chromato_hidden_dim=32,
            fusion_hidden_dims=[512, 256, 128],
            dropout_rate=0.0,
            use_attention=False,
            use_gating=False,
            feature_level=feature_level
        )
        
        # 根据特征级别确定要传入的特征
        if feature_level == 'graph':
            p_features, t_features, c_features = None, None, None
        elif feature_level == 'graph_physchem':
            p_features, t_features, c_features = physchem_features, None, None
        elif feature_level == 'graph_physchem_toxicity':
            p_features, t_features, c_features = physchem_features, toxicity_features, None
        elif feature_level == 'all':
            p_features, t_features, c_features = physchem_features, toxicity_features, chromato_features
        
        # 运行基准测试
        param_count, param_memory, peak_memory, avg_time = benchmark_model(
            model, z, pos, batch, p_features, t_features, c_features, iterations=10
        )
        
        # 估算150轮的时间
        estimated_150_time = avg_time * 150
        
        # 记录结果
        result = {
            'feature_level': feature_level,
            'param_count_m': param_count / 1e6,  # 百万参数量
            'param_memory_mb': param_memory / (1024 * 1024),
            'peak_memory_mb': peak_memory / (1024 * 1024),
            'avg_time_ms': avg_time * 1000,
            'estimated_150_time_s': estimated_150_time
        }
        results.append(result)
        
        print(f"  模型参数量: {result['param_count_m']:.2f}M")
        print(f"  参数内存占用: {result['param_memory_mb']:.2f} MB")
        print(f"  峰值内存占用: {result['peak_memory_mb']:.2f} MB")
        print(f"  平均单轮时间: {result['avg_time_ms']:.2f} ms")
        print(f"  150轮估算时间: {result['estimated_150_time_s']:.2f} s")
        
        # 清理
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # 输出汇总结果
    print("\n" + "="*100)
    print("汇总结果:")
    print("="*100)
    print(f"{'特征级别':<25} {'参数量(M)':<15} {'参数内存(MB)':<15} {'峰值内存(MB)':<15} {'单轮时间(ms)':<15} {'150轮时间(s)':<15}")
    print("-"*100)
    
    for result in results:
        print(f"{result['feature_level']:<25} "
              f"{result['param_count_m']:<15.2f} "
              f"{result['param_memory_mb']:<15.2f} "
              f"{result['peak_memory_mb']:<15.2f} "
              f"{result['avg_time_ms']:<15.2f} "
              f"{result['estimated_150_time_s']:<15.2f}")
    
    # 保存结果到文件
    with open('visnet_v2_benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存到 visnet_v2_benchmark_results.json")


if __name__ == "__main__":
    main()