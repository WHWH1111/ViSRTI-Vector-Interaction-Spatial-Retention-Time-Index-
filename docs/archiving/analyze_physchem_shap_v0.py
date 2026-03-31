#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import json
import torch
import numpy as np
import argparse
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import time
from datetime import datetime

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 导入模型
from models.visnet_v2 import VisNetV2

# 导入工具函数
from core import load_datasets, preprocess_visnet_data

# 导入自定义可视化函数
from utils.shap.draw import visualize_shap_results as custom_visualize_shap_results


def load_model(model_path, training_params_path, device):
    """
    加载训练好的VisNetV2模型
    """
    # 读取训练参数
    with open(training_params_path, 'r') as f:
        training_params = json.load(f)
    
    # 创建VisNetV2模型实例
    model = VisNetV2(
        node_feature_dim=training_params.get('visnet_v2_node_feature_dim', 64),
        physchem_feature_dim=training_params.get('visnet_v2_physchem_feature_dim', 4),
        toxicity_feature_dim=training_params.get('visnet_v2_toxicity_feature_dim', 4),
        chromato_feature_dim=training_params.get('visnet_v2_chromato_feature_dim', 2),
        graph_hidden_dim=training_params.get('visnet_v2_graph_hidden_dim', 512),
        physchem_hidden_dim=training_params.get('visnet_v2_physchem_hidden_dim', 64),
        toxicity_hidden_dim=training_params.get('visnet_v2_toxicity_hidden_dim', 64),
        chromato_hidden_dim=training_params.get('visnet_v2_chromato_hidden_dim', 32),
        fusion_hidden_dims=training_params.get('visnet_v2_fusion_hidden_dims', [512, 256, 128]),
        dropout_rate=training_params.get('dropout', 0.0),  # 设置为0以确保确定性
        use_attention=training_params.get('visnet_v2_use_attention', False),
        use_gating=training_params.get('visnet_v2_use_gating', False)
    ).to(device)
    
    # 加载模型权重，忽略不匹配的键
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # 确保所有BatchNorm层都在eval模式下
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.eval()
            # 对于小批量或单样本，使用训练时的统计信息
            module.track_running_stats = True
            
    return model


class PhysChemSHAPWrapper:
    """
    包装模型用于SHAP分析，固定图结构，只变化物化属性
    """
    def __init__(self, model, fixed_graph_data, device, batch_size=32):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.fixed_z = fixed_graph_data['z'].to(device)
        self.fixed_pos = fixed_graph_data['pos'].to(device)
        self.fixed_batch = fixed_graph_data['batch'].to(device) if 'batch' in fixed_graph_data else None
        
        # 如果没有提供batch，则创建一个默认的batch索引
        if self.fixed_batch is None:
            self.fixed_batch = torch.zeros(self.fixed_z.shape[0], dtype=torch.long, device=device)
    
    def predict(self, physchem_features):
        """
        预测函数，用于SHAP分析
        
        Args:
            physchem_features: 物化特征数组 [n_samples, 4]
            
        Returns:
            predictions: 预测结果 [n_samples]
        """
        # 确保输入是正确的类型和设备
        if isinstance(physchem_features, np.ndarray):
            physchem_tensor = torch.FloatTensor(physchem_features).to(self.device)
        else:
            physchem_tensor = physchem_features.to(self.device)
        
        # 获取样本数量
        n_samples = physchem_tensor.shape[0]
        
        # 确保至少有2个样本以满足BatchNorm要求
        if n_samples == 1:
            # 复制样本以满足最小批量大小要求
            physchem_tensor = torch.cat([physchem_tensor, physchem_tensor], dim=0)
            duplicate_prediction = True
        else:
            duplicate_prediction = False
        
        # 分批处理预测以节省内存
        predictions = []
        n_samples_actual = physchem_tensor.shape[0]
        
        # 为图结构数据计算基础信息
        n_atoms = self.fixed_z.shape[0]
        
        # 分批处理
        for i in range(0, n_samples_actual, self.batch_size):
            batch_end = min(i + self.batch_size, n_samples_actual)
            current_batch_size = batch_end - i
            
            # 获取当前批次的物化特征
            current_physchem = physchem_tensor[i:batch_end]
            
            # 扩展图数据以匹配当前批次大小
            z = self.fixed_z.repeat(current_batch_size)
            pos = self.fixed_pos.repeat(current_batch_size, 1, 1).view(-1, 3)
            batch = torch.arange(current_batch_size, device=self.device).repeat_interleave(n_atoms)
            
            with torch.no_grad():
                pred, _ = self.model(
                    z=z, pos=pos, batch=batch,
                    physchem_features=current_physchem
                )
            
            predictions.append(pred)
        
        # 合并所有批次的预测结果
        final_pred = torch.cat(predictions, dim=0)
        
        # 处理重复样本的情况
        if duplicate_prediction:
            final_pred = final_pred[:1]  # 只取第一个预测结果
        
        # 确保输出是一维的
        if final_pred.ndim > 1:
            final_pred = final_pred.squeeze()
        
        return final_pred.cpu().numpy()


def prepare_physchem_data(dataset, preprocessed_data, sample_size=100):
    """
    准备物化特征数据用于SHAP分析
    
    Args:
        dataset: 数据集
        preprocessed_data: 预处理后的数据
        sample_size: 采样数量
        
    Returns:
        physchem_features: 物化特征数组
        smiles_list: 对应的SMILES列表
    """
    # 随机选择样本
    indices = np.random.choice(len(dataset), min(sample_size, len(dataset)), replace=False)
    
    # 提取物化特征
    physchem_features = []
    valid_indices = []
    smiles_list = []
    
    for i in indices:
        smiles = dataset[i][0]  # 假设SMILES在第一个位置
        if smiles in preprocessed_data:
            data_entry = preprocessed_data[smiles]
            if 'physchem_features' in data_entry and data_entry['physchem_features'] is not None:
                # 确保是torch.Tensor或numpy数组类型
                feat = data_entry['physchem_features']
                if isinstance(feat, list):
                    feat = np.array(feat)
                elif isinstance(feat, torch.Tensor):
                    feat = feat.numpy()
                
                physchem_features.append(feat)
                valid_indices.append(i)
                smiles_list.append(smiles)
    
    if len(physchem_features) == 0:
        raise ValueError("没有找到有效的物化特征数据")
    
    physchem_features = np.stack(physchem_features)
    print(f"使用 {len(physchem_features)} 个样本进行SHAP分析")
    
    return physchem_features, smiles_list


def prepare_combined_physchem_data(dev_dataset, test_dataset, dev_preprocessed_data, test_preprocessed_data, sample_size=100):
    """
    准备合并的开发集和测试集物化特征数据用于SHAP分析
    
    Args:
        dev_dataset: 开发数据集
        test_dataset: 测试数据集
        dev_preprocessed_data: 开发集预处理后的数据
        test_preprocessed_data: 测试集预处理后的数据
        sample_size: 采样数量
        
    Returns:
        physchem_features: 物化特征数组
        smiles_list: 对应的SMILES列表
    """
    # 合并两个数据集的预处理数据
    combined_preprocessed_data = {**dev_preprocessed_data, **test_preprocessed_data}
    
    # 合并两个数据集的SMILES
    dev_smiles_list = [dev_dataset[i][0] for i in range(len(dev_dataset))]
    test_smiles_list = [test_dataset[i][0] for i in range(len(test_dataset))]
    combined_smiles_list = dev_smiles_list + test_smiles_list
    
    # 随机选择样本
    total_samples = len(combined_smiles_list)
    actual_sample_size = min(sample_size, total_samples)
    selected_smiles = np.random.choice(combined_smiles_list, actual_sample_size, replace=False)
    
    # 提取物化特征
    physchem_features = []
    smiles_list = []
    
    for smiles in selected_smiles:
        if smiles in combined_preprocessed_data:
            data_entry = combined_preprocessed_data[smiles]
            if 'physchem_features' in data_entry and data_entry['physchem_features'] is not None:
                # 确保是torch.Tensor或numpy数组类型
                feat = data_entry['physchem_features']
                if isinstance(feat, list):
                    feat = np.array(feat)
                elif isinstance(feat, torch.Tensor):
                    feat = feat.numpy()
                
                physchem_features.append(feat)
                smiles_list.append(smiles)
    
    if len(physchem_features) == 0:
        raise ValueError("没有找到有效的物化特征数据")
    
    physchem_features = np.stack(physchem_features)
    print(f"使用 {len(physchem_features)} 个样本进行SHAP分析 (来自dev和test集合并数据)")
    
    return physchem_features, smiles_list


def compute_shap_values(model, physchem_features, reference_graph_data, device, background_samples=50, batch_size=32):
    """
    计算物化特征的SHAP值
    
    Args:
        model: 训练好的模型
        physchem_features: 物化特征数组 [n_samples, 4]
        reference_graph_data: 参考图结构数据
        device: 计算设备
        background_samples: 背景样本数量
        batch_size: 批处理大小
        
    Returns:
        shap_values: SHAP值数组
        explainer: SHAP解释器
    """
    # 创建包装模型
    wrapped_model = PhysChemSHAPWrapper(model, reference_graph_data, device, batch_size)
    
    # 选择背景数据（用于SHAP参考）
    background_data = physchem_features[:min(background_samples, len(physchem_features))]
    
    # 创建SHAP解释器
    print("创建SHAP解释器...")
    explainer = shap.KernelExplainer(wrapped_model.predict, background_data)
    
    # 计算SHAP值
    print("计算SHAP值...")
    shap_values = explainer.shap_values(physchem_features)
    
    return shap_values, explainer




def analyze_feature_importance(shap_values, feature_names, output_dir):
    """
    分析并保存特征重要性结果
    
    Args:
        shap_values: SHAP值数组
        feature_names: 特征名称列表
        output_dir: 输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算特征重要性
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    
    # 创建特征重要性字典
    feature_importance = {}
    for i, name in enumerate(feature_names):
        feature_importance[name] = float(mean_abs_shap[i])
    
    # 按重要性排序
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    # 打印结果
    print("\n特征重要性排序 (基于平均|SHAP值|):")
    for i, (name, importance) in enumerate(sorted_features):
        print(f"{i+1}. {name}: {importance:.6f}")
    
    # 保存特征重要性到JSON文件
    with open(os.path.join(output_dir, 'feature_importance.json'), 'w') as f:
        json.dump(feature_importance, f, indent=2)
    
    # 保存详细SHAP值到文件
    shap_details = {
        'feature_names': feature_names,
        'mean_abs_shap': mean_abs_shap.tolist(),
        'feature_ranking': sorted_features
    }
    
    with open(os.path.join(output_dir, 'shap_analysis_results.json'), 'w') as f:
        json.dump(shap_details, f, indent=2)
    
    return sorted_features


def main():
    parser = argparse.ArgumentParser(description='VisNetV2 SHAP Analysis for Physicochemical Features')
    
    # 数据和模型参数
    parser.add_argument('--model-path', type=str, 
                        default='./log/neg-train_20251028-110803_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/model.pt',
                        help='训练好的模型路径')
    parser.add_argument('--training-params-path', type=str,
                        default='./log/neg-train_20251028-110803_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/training_params.json',
                        help='训练参数文件路径')
    parser.add_argument('--data-path', type=str, default='./data/MMF-3/',
                        help='数据路径')
    parser.add_argument('--dataset-name', type=str, default='MMF_GNN_neg',
                        help='数据集名称')
    parser.add_argument('--output-dir', type=str, default='./shap_analysis_results',
                        help='SHAP结果输出目录')
    parser.add_argument('--sample-size', type=int, default=500,
                        help='用于SHAP分析的样本数量')
    parser.add_argument('--background-samples', type=int, default=100,
                        help='用于SHAP背景的样本数量')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='SHAP分析的批处理大小')
    parser.add_argument('--use-combined-data', action='store_true',
                        help='是否使用dev和test合并的数据，默认只使用test数据')
    parser.add_argument('--use-train-data', action='store_true',
                        help='是否使用train数据，默认只使用test数据')
    
    args = parser.parse_args()
    
    # 创建带时间戳的输出目录
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_output_dir = args.output_dir
    output_dir_with_timestamp = os.path.join(base_output_dir, f"shap_analysis_{timestamp}")
    os.makedirs(output_dir_with_timestamp, exist_ok=True)
    
    # 保存命令行参数到日志文件
    args_log_path = os.path.join(output_dir_with_timestamp, 'shap_analysis_args.json')
    with open(args_log_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 记录开始时间
    start_time = time.time()
    
    # 加载模型
    print("加载模型...")
    model = load_model(args.model_path, args.training_params_path, device)
    
    # 准备数据
    print("准备数据...")
    
    # 模拟训练参数以匹配数据加载需求
    class Args:
        model = 'visnet_v2'
        visnet_v2_feature_level = 'graph_physchem'  # 只使用图和物化特征
        debug_size = None
    
    args_obj = Args()
    
    # 加载数据集
    dataset_train, dataset_dev, dataset_test, _, _, _ = load_datasets(
        args_obj, args.data_path, args.dataset_name, max_data=None)
    
    # VisNet系列模型需要特殊的预处理
    train_preprocessed_data, dev_preprocessed_data, test_preprocessed_data, \
    dataset_train_filtered, dataset_dev_filtered, dataset_test_filtered = preprocess_visnet_data(
        args_obj, dataset_train, dataset_dev, dataset_test, args.dataset_name, "visnet_train_", "visnet_test_")
    
    # 准备物化特征数据
    if args.use_combined_data:
        print("使用dev和test合并的数据集")
        physchem_features, smiles_list = prepare_combined_physchem_data(
            dataset_dev_filtered, dataset_test_filtered, dev_preprocessed_data, test_preprocessed_data, args.sample_size)
    elif args.use_train_data:
        print("使用train数据集")
        physchem_features, smiles_list = prepare_physchem_data(
            dataset_train_filtered, train_preprocessed_data, args.sample_size)
    else:
        print("使用test数据集")
        physchem_features, smiles_list = prepare_physchem_data(
            dataset_test_filtered, test_preprocessed_data, args.sample_size)
    
    # 获取一个样本的图数据作为固定输入
    sample_smiles = smiles_list[0]
    # 根据SMILES来源决定使用哪个预处理数据集
    if sample_smiles in train_preprocessed_data:
        reference_graph_data = train_preprocessed_data[sample_smiles]
        print(f"参考图数据来自train集: {sample_smiles}")
    elif sample_smiles in dev_preprocessed_data:
        reference_graph_data = dev_preprocessed_data[sample_smiles]
        print(f"参考图数据来自dev集: {sample_smiles}")
    elif sample_smiles in test_preprocessed_data:
        reference_graph_data = test_preprocessed_data[sample_smiles]
        print(f"参考图数据来自test集: {sample_smiles}")
    else:
        # 默认使用test集的第一个样本
        sample_smiles = dataset_test_filtered[0][0]
        reference_graph_data = test_preprocessed_data[sample_smiles]
        print(f"参考图数据来自test集(默认): {sample_smiles}")
    
    # 计算SHAP值
    print("开始SHAP分析...")
    shap_values, explainer = compute_shap_values(
        model, physchem_features, reference_graph_data, device, args.background_samples, args.batch_size)
    
    # 物化特征名称
    feature_names = ['Monoiso_Mass', 'LogKow', 'LogP', 'Koc_predicted']
    
    # 分析特征重要性
    print("分析特征重要性...")
    feature_importance = analyze_feature_importance(shap_values, feature_names, output_dir_with_timestamp)
    
    # 使用自定义可视化函数
    print("生成可视化结果...")
    custom_visualize_shap_results(shap_values, physchem_features, feature_names, output_dir_with_timestamp)
    
    # 记录运行时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 保存运行信息
    runtime_info = {
        'start_time': datetime.fromtimestamp(start_time).isoformat(),
        'end_time': datetime.fromtimestamp(end_time).isoformat(),
        'elapsed_time_seconds': elapsed_time,
        'samples_used': len(physchem_features),
        'background_samples': args.background_samples,
        'batch_size': args.batch_size,
        'use_combined_data': args.use_combined_data
    }
    
    runtime_info_path = os.path.join(output_dir_with_timestamp, 'runtime_info.json')
    with open(runtime_info_path, 'w') as f:
        json.dump(runtime_info, f, indent=2)
    
    print(f"\nSHAP分析完成，结果保存在 {output_dir_with_timestamp}")
    print(f"总耗时: {elapsed_time:.2f} 秒")
    print("\n特征重要性排序:")
    for i, (name, importance) in enumerate(feature_importance):
        print(f"{i+1}. {name}: {importance:.6f}")


if __name__ == "__main__":
    main()