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
    
    # 获取特征掩码
    physchem_mask = training_params.get('visnet_v2_physchem_mask', None)
    toxicity_mask = training_params.get('visnet_v2_toxicity_mask', None)
    chromato_mask = training_params.get('visnet_v2_chromato_mask', None)
    
    # 创建VisNetV2模型实例，使用固定的原始特征维度，并传递掩码参数
    model = VisNetV2(
        node_feature_dim=training_params.get('visnet_v2_node_feature_dim', 64),
        physchem_feature_dim=4,  # 使用原始维度
        toxicity_feature_dim=4,  # 使用原始维度
        chromato_feature_dim=2,  # 使用原始维度
        graph_hidden_dim=training_params.get('visnet_v2_graph_hidden_dim', 512),
        physchem_hidden_dim=training_params.get('visnet_v2_physchem_hidden_dim', 128),
        toxicity_hidden_dim=training_params.get('visnet_v2_toxicity_hidden_dim', 64),
        toxicity_intermediate_dim=training_params.get('visnet_v2_toxicity_intermediate_dim', 32),
        chromato_hidden_dim=training_params.get('visnet_v2_chromato_hidden_dim', 32),
        fusion_hidden_dims=training_params.get('visnet_v2_fusion_hidden_dims', [512, 256, 128]),
        dropout_rate=training_params.get('dropout', 0.0),  # 设置为0以确保确定性
        use_attention=training_params.get('visnet_v2_use_attention', False),
        use_gating=training_params.get('visnet_v2_use_gating', False),
        physchem_feature_mask=physchem_mask,      # 传递物理化学特征掩码
        toxicity_feature_mask=toxicity_mask,      # 传递毒性特征掩码
        chromato_feature_mask=chromato_mask       # 传递色谱特征掩码
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
    
    # 保存标准化信息和掩码信息到模型对象中，供SHAP分析使用
    model.standardization_info = training_params.get('standardization_info', {})
    model.feature_level = training_params.get('visnet_v2_feature_level', 'graph_physchem')
    model.physchem_mask = physchem_mask
    model.toxicity_mask = toxicity_mask
    model.chromato_mask = chromato_mask
    model.original_physchem_dim = 4
    model.original_toxicity_dim = 4
    model.original_chromato_dim = 2
    
    return model


class MultiFeatureSHAPWrapper:
    """
    包装模型用于SHAP分析，固定图结构，只变化多种特征属性
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
        
        # 获取标准化参数
        self.standardization_info = getattr(model, 'standardization_info', {})
        full_physchem_mean = np.array(self.standardization_info.get('physchem', {}).get('mean', [0, 0, 0, 0]))
        full_physchem_std = np.array(self.standardization_info.get('physchem', {}).get('std', [1, 1, 1, 1]))
        full_toxicity_mean = np.array(self.standardization_info.get('toxicity', {}).get('mean', [0, 0, 0, 0]))
        full_toxicity_std = np.array(self.standardization_info.get('toxicity', {}).get('std', [1, 1, 1, 1]))
        full_chromato_mean = np.array(self.standardization_info.get('chromato', {}).get('mean', [0, 0]))
        full_chromato_std = np.array(self.standardization_info.get('chromato', {}).get('std', [1, 1]))
        
        self.physchem_mask = getattr(model, 'physchem_mask', None)
        self.toxicity_mask = getattr(model, 'toxicity_mask', None)
        self.chromato_mask = getattr(model, 'chromato_mask', None)
        
        # 根据特征掩码过滤mean和std
        if self.physchem_mask is not None:
            mask_array = np.array(self.physchem_mask, dtype=bool)
            self.full_physchem_mean = full_physchem_mean[mask_array]
            self.full_physchem_std = full_physchem_std[mask_array]
        else:
            self.full_physchem_mean = full_physchem_mean
            self.full_physchem_std = full_physchem_std
            
        if self.toxicity_mask is not None:
            mask_array = np.array(self.toxicity_mask, dtype=bool)
            self.full_toxicity_mean = full_toxicity_mean[mask_array]
            self.full_toxicity_std = full_toxicity_std[mask_array]
        else:
            self.full_toxicity_mean = full_toxicity_mean
            self.full_toxicity_std = full_toxicity_std
            
        if self.chromato_mask is not None:
            mask_array = np.array(self.chromato_mask, dtype=bool)
            self.full_chromato_mean = full_chromato_mean[mask_array]
            self.full_chromato_std = full_chromato_std[mask_array]
        else:
            self.full_chromato_mean = full_chromato_mean
            self.full_chromato_std = full_chromato_std
        
        # 获取模型使用的特征级别和掩码
        self.feature_level = getattr(model, 'feature_level', 'graph_physchem')
        
        # 获取原始维度和实际维度
        self.original_physchem_dim = min(4, sum(self.physchem_mask) if self.physchem_mask is not None else 4)
        self.original_toxicity_dim = min(4, sum(self.toxicity_mask) if self.toxicity_mask is not None else 4)
        self.original_chromato_dim = min(2, sum(self.chromato_mask) if self.chromato_mask is not None else 2)
        
    def predict(self, features):
        """
        预测函数，用于SHAP分析
        
        Args:
            features: 特征数组，根据模型的feature_level决定包含哪些特征
                  - graph_physchem_toxicity: [n_samples, physchem_dim+toxicity_dim]
                  - all: [n_samples, physchem_dim+toxicity_dim+chromato_dim]
                  - graph_physchem: [n_samples, physchem_dim]
            
        Returns:
            predictions: 预测结果 [n_samples]
        """
        # 确保输入是正确的类型和设备
        if isinstance(features, np.ndarray):
            features_tensor = torch.FloatTensor(features).to(self.device)
        else:
            features_tensor = features.to(self.device)
        
        # 获取样本数量
        n_samples = features_tensor.shape[0]
        
        # 确保至少有2个样本以满足BatchNorm要求
        if n_samples == 1:
            # 复制样本以满足最小批量大小要求
            features_tensor = torch.cat([features_tensor, features_tensor], dim=0)
            duplicate_prediction = True
        else:
            duplicate_prediction = False
        
        # 分批处理预测以节省内存
        predictions = []
        n_samples_actual = features_tensor.shape[0]
        
        # 为图结构数据计算基础信息
        n_atoms = self.fixed_z.shape[0]
        
        # 分批处理
        for i in range(0, n_samples_actual, self.batch_size):
            batch_end = min(i + self.batch_size, n_samples_actual)
            current_batch_size = batch_end - i
            
            # 获取当前批次的特征
            current_features = features_tensor[i:batch_end]
            
            # 根据模型使用的特征级别分离特征
            if self.feature_level == 'all':  # 物化+毒性+色谱
                # 传递完整的特征给模型，让模型内部处理掩码
                current_physchem = current_features[:, :self.original_physchem_dim]
                current_toxicity = current_features[:, self.original_physchem_dim:self.original_physchem_dim+self.original_toxicity_dim]
                current_chromato = current_features[:, self.original_physchem_dim+self.original_toxicity_dim:]
                
                # 如果模型使用了特征标准化，则对输入特征进行标准化处理
                if len(self.standardization_info) > 0:
                    # 标准化时使用完整的特征和完整的标准化参数
                    current_physchem = (current_physchem - torch.FloatTensor(self.full_physchem_mean).to(self.device)) / torch.FloatTensor(self.full_physchem_std).to(self.device)
                    current_toxicity = (current_toxicity - torch.FloatTensor(self.full_toxicity_mean).to(self.device)) / torch.FloatTensor(self.full_toxicity_std).to(self.device)
                    current_chromato = (current_chromato - torch.FloatTensor(self.full_chromato_mean).to(self.device)) / torch.FloatTensor(self.full_chromato_std).to(self.device)
                    
            elif self.feature_level == 'graph_physchem_toxicity':  # 物化+毒性

                # 传递完整的特征给模型，让模型内部处理掩码
                current_physchem = current_features[:, :self.original_physchem_dim]
                current_toxicity = current_features[:, self.original_physchem_dim:self.original_physchem_dim+self.original_toxicity_dim]
                current_chromato = None
                
                # 如果模型使用了特征标准化，则对输入特征进行标准化处理
                if len(self.standardization_info) > 0:
                    # 标准化时使用完整的特征和完整的标准化参数
                    current_physchem = (current_physchem - torch.FloatTensor(self.full_physchem_mean).to(self.device)) / torch.FloatTensor(self.full_physchem_std).to(self.device)
                    current_toxicity = (current_toxicity - torch.FloatTensor(self.full_toxicity_mean).to(self.device)) / torch.FloatTensor(self.full_toxicity_std).to(self.device)
            else:  # 默认: 物化
                # 传递完整的特征给模型，让模型内部处理掩码
                current_physchem = current_features
                current_toxicity = None
                current_chromato = None
                
                # 如果模型使用了特征标准化，则对输入特征进行标准化处理
                if len(self.standardization_info) > 0:
                    # 标准化时使用完整的特征和完整的标准化参数
                    current_physchem = (current_physchem - torch.FloatTensor(self.full_physchem_mean).to(self.device)) / torch.FloatTensor(self.full_physchem_std).to(self.device)
            
            # 扩展图数据以匹配当前批次大小
            z = self.fixed_z.repeat(current_batch_size)
            pos = self.fixed_pos.repeat(current_batch_size, 1, 1).view(-1, 3)
            batch = torch.arange(current_batch_size, device=self.device).repeat_interleave(n_atoms)
            
            with torch.no_grad():
                # 临时设置模型为评估模式，确保BatchNorm层正常工作
                was_training = self.model.training
                if was_training:
                    self.model.eval()
                
                # 如果批次大小为1，需要特殊处理BatchNorm层
                batch_norm_layers = []
                if current_batch_size == 1:
                    # 对于单样本情况，我们需要临时修改模型中的BatchNorm层
                    # 使其使用训练时的统计信息而不是当前批次的统计信息
                    for module in self.model.modules():
                        if isinstance(module, nn.BatchNorm1d):
                            # 保存原始状态
                            batch_norm_layers.append({
                                'layer': module,
                                'training': module.training,
                                'track_running_stats': module.track_running_stats
                            })
                            # 设置为评估模式
                            module.eval()
                            module.track_running_stats = True
                
                pred, _ = self.model(
                    z=z, pos=pos, batch=batch,
                    physchem_features=current_physchem,
                    toxicity_features=current_toxicity,
                    chromato_features=current_chromato
                )
                
                # 恢复BatchNorm层的原始状态
                for bn_info in batch_norm_layers:
                    bn_layer = bn_info['layer']
                    bn_layer.train(bn_info['training']) if bn_info['training'] else bn_layer.eval()
                    bn_layer.track_running_stats = bn_info['track_running_stats']
                
                # 恢复模型的原始训练状态
                if was_training:
                    self.model.train()
            
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


def prepare_multi_features_data(dataset, preprocessed_data, feature_level, physchem_mask=None, toxicity_mask=None, chromato_mask=None, sample_size=100):
    """
    准备多种特征数据用于SHAP分析
    
    Args:
        dataset: 数据集
        preprocessed_data: 预处理后的数据
        feature_level: 特征级别 ('graph_physchem', 'graph_physchem_toxicity', 'all')
        physchem_mask: 物化特征掩码
        toxicity_mask: 毒性特征掩码
        chromato_mask: 色谱特征掩码
        sample_size: 采样数量
        
    Returns:
        multi_features: 多种特征数组
        smiles_list: 对应的SMILES列表
        feature_names: 特征名称列表
    """
    # 随机选择样本
    indices = np.random.choice(len(dataset), min(sample_size, len(dataset)), replace=False)
    
    # 提取特征
    multi_features = []
    smiles_list = []
    
    for i in indices:
        smiles = dataset[i][0]  # 假设SMILES在第一个位置
        if smiles in preprocessed_data:
            data_entry = preprocessed_data[smiles]
            
            # 根据特征级别收集特征
            features = []
            
            # 物化特征 (总是存在)
            if 'physchem_features' in data_entry and data_entry['physchem_features'] is not None:
                feat = data_entry['physchem_features']
                if isinstance(feat, list):
                    feat = np.array(feat)
                elif isinstance(feat, torch.Tensor):
                    feat = feat.numpy()
                
                # 应用掩码
                if physchem_mask is not None:
                    mask_array = np.array(physchem_mask, dtype=bool)
                    feat = feat[mask_array]
                features.append(feat)
            else:
                continue  # 如果没有物化特征，跳过这个样本
            
            # 毒性特征 (如果需要)
            if feature_level in ['graph_physchem_toxicity', 'all']:
                if 'toxicity_features' in data_entry and data_entry['toxicity_features'] is not None:
                    feat = data_entry['toxicity_features']
                    if isinstance(feat, list):
                        feat = np.array(feat)
                    elif isinstance(feat, torch.Tensor):
                        feat = feat.numpy()
                    
                    # 应用掩码
                    if toxicity_mask is not None:
                        mask_array = np.array(toxicity_mask, dtype=bool)
                        feat = feat[mask_array]
                    features.append(feat)
                else:
                    continue  # 如果需要毒性特征但不存在，跳过这个样本
            
            # 色谱特征 (如果需要)
            if feature_level == 'all':
                if 'chromato_features' in data_entry and data_entry['chromato_features'] is not None:
                    feat = data_entry['chromato_features']
                    if isinstance(feat, list):
                        feat = np.array(feat)
                    elif isinstance(feat, torch.Tensor):
                        feat = feat.numpy()
                    
                    # 应用掩码
                    if chromato_mask is not None:
                        mask_array = np.array(chromato_mask, dtype=bool)
                        feat = feat[mask_array]
                    features.append(feat)
                else:
                    continue  # 如果需要色谱特征但不存在，跳过这个样本
            
            # 合并特征
            combined_features = np.concatenate(features)
            multi_features.append(combined_features)
            smiles_list.append(smiles)
    
    if len(multi_features) == 0:
        raise ValueError("没有找到有效的特征数据")
    
    multi_features = np.stack(multi_features)
    print(f"使用 {len(multi_features)} 个样本进行SHAP分析")
    
    # 构建特征名称列表
    feature_names = []
    
    # 物化特征名称
    physchem_names = ['Monoiso_Mass', 'LogKow', 'LogP', 'Koc_predicted (L/kg)']
    if physchem_mask is not None:
        mask_array = np.array(physchem_mask, dtype=bool)
        physchem_names = [name for i, name in enumerate(physchem_names) if mask_array[i]]
    feature_names.extend(physchem_names)
    
    # 毒性特征名称 (如果需要)
    if feature_level in ['graph_physchem_toxicity', 'all']:
        toxicity_names = ['Tetrahymena_pyriformis_toxicity', 'Daphnia_toxicity', 'Algae_toxicity', 'Pimephales_promelas_toxicity']
        if toxicity_mask is not None:
            mask_array = np.array(toxicity_mask, dtype=bool)
            toxicity_names = [name for i, name in enumerate(toxicity_names) if mask_array[i]]
        feature_names.extend(toxicity_names)
    
    # 色谱特征名称 (如果需要)
    if feature_level == 'all':
        chromato_names = ['Prob. +ESI', 'Prob. -ESI']
        if chromato_mask is not None:
            mask_array = np.array(chromato_mask, dtype=bool)
            chromato_names = [name for i, name in enumerate(chromato_names) if mask_array[i]]
        feature_names.extend(chromato_names)
    
    return multi_features, smiles_list, feature_names


def prepare_combined_multi_features_data(dev_dataset, test_dataset, dev_preprocessed_data, test_preprocessed_data, feature_level, physchem_mask=None, toxicity_mask=None, chromato_mask=None, sample_size=100):
    """
    准备合并的开发集和测试集多种特征数据用于SHAP分析
    
    Args:
        dev_dataset: 开发数据集
        test_dataset: 测试数据集
        dev_preprocessed_data: 开发集预处理后的数据
        test_preprocessed_data: 测试集预处理后的数据
        feature_level: 特征级别 ('graph_physchem', 'graph_physchem_toxicity', 'all')
        physchem_mask: 物化特征掩码
        toxicity_mask: 毒性特征掩码
        chromato_mask: 色谱特征掩码
        sample_size: 采样数量
        
    Returns:
        multi_features: 多种特征数组
        smiles_list: 对应的SMILES列表
        feature_names: 特征名称列表
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
    
    # 提取特征
    multi_features = []
    smiles_list = []
    
    for smiles in selected_smiles:
        if smiles in combined_preprocessed_data:
            data_entry = combined_preprocessed_data[smiles]
            
            # 根据特征级别收集特征
            features = []
            
            # 物化特征 (总是存在)
            if 'physchem_features' in data_entry and data_entry['physchem_features'] is not None:
                feat = data_entry['physchem_features']
                if isinstance(feat, list):
                    feat = np.array(feat)
                elif isinstance(feat, torch.Tensor):
                    feat = feat.numpy()
                
                # 应用掩码
                if physchem_mask is not None:
                    mask_array = np.array(physchem_mask, dtype=bool)
                    feat = feat[mask_array]
                features.append(feat)
            else:
                continue  # 如果没有物化特征，跳过这个样本
            
            # 毒性特征 (如果需要)
            if feature_level in ['graph_physchem_toxicity', 'all']:
                if 'toxicity_features' in data_entry and data_entry['toxicity_features'] is not None:
                    feat = data_entry['toxicity_features']
                    if isinstance(feat, list):
                        feat = np.array(feat)
                    elif isinstance(feat, torch.Tensor):
                        feat = feat.numpy()
                    
                    # 应用掩码
                    if toxicity_mask is not None:
                        mask_array = np.array(toxicity_mask, dtype=bool)
                        feat = feat[mask_array]
                    features.append(feat)
                else:
                    continue  # 如果需要毒性特征但不存在，跳过这个样本
            
            # 色谱特征 (如果需要)
            if feature_level == 'all':
                if 'chromato_features' in data_entry and data_entry['chromato_features'] is not None:
                    feat = data_entry['chromato_features']
                    if isinstance(feat, list):
                        feat = np.array(feat)
                    elif isinstance(feat, torch.Tensor):
                        feat = feat.numpy()
                    
                    # 应用掩码
                    if chromato_mask is not None:
                        mask_array = np.array(chromato_mask, dtype=bool)
                        feat = feat[mask_array]
                    features.append(feat)
                else:
                    continue  # 如果需要色谱特征但不存在，跳过这个样本
            
            # 合并特征
            combined_features = np.concatenate(features)
            multi_features.append(combined_features)
            smiles_list.append(smiles)
    
    if len(multi_features) == 0:
        raise ValueError("没有找到有效的特征数据")
    
    multi_features = np.stack(multi_features)
    print(f"使用 {len(multi_features)} 个样本进行SHAP分析 (来自dev和test集合并数据)")
    
    # 构建特征名称列表
    feature_names = []
    
    # 物化特征名称 & mask 过滤
    physchem_names = ['Monoiso_Mass', 'LogKow', 'LogP', 'Koc_predicted (L/kg)']
    if physchem_mask is not None:
        mask_array = np.array(physchem_mask, dtype=bool)
        physchem_names = [name for i, name in enumerate(physchem_names) if mask_array[i]]
    feature_names.extend(physchem_names)
    
    # 毒性特征名称 (如果需要)
    if feature_level in ['graph_physchem_toxicity', 'all']:
        toxicity_names = ['Tetrahymena_pyriformis_toxicity', 'Daphnia_toxicity', 'Algae_toxicity', 'Pimephales_promelas_toxicity']
        if toxicity_mask is not None:
            mask_array = np.array(toxicity_mask, dtype=bool)
            toxicity_names = [name for i, name in enumerate(toxicity_names) if mask_array[i]]
        feature_names.extend(toxicity_names)
    
    # 色谱特征名称 (如果需要)
    if feature_level == 'all':
        chromato_names = ['Prob. +ESI', 'Prob. -ESI']
        if chromato_mask is not None:
            mask_array = np.array(chromato_mask, dtype=bool)
            chromato_names = [name for i, name in enumerate(chromato_names) if mask_array[i]]
        feature_names.extend(chromato_names)
    
    return multi_features, smiles_list, feature_names


def compute_shap_values(model, features, reference_graph_data, device, background_samples=50, batch_size=32):
    """
    计算多种特征的SHAP值
    
    Args:
        model: 训练好的模型
        features: 特征数组
        reference_graph_data: 参考图结构数据
        device: 计算设备
        background_samples: 背景样本数量
        batch_size: 批处理大小
        
    Returns:
        shap_values: SHAP值数组
        explainer: SHAP解释器
    """
    # 获取标准化参数
    standardization_info = getattr(model, 'standardization_info', {})
    feature_level = getattr(model, 'feature_level', 'graph_physchem')
    
    # 创建包装模型
    wrapped_model = MultiFeatureSHAPWrapper(model, reference_graph_data, device, batch_size)
    
    # 选择背景数据（用于SHAP参考）
    background_data = features[:min(background_samples, len(features))]
    
    # 创建SHAP解释器
    print("创建SHAP解释器...")
    explainer = shap.KernelExplainer(wrapped_model.predict, background_data)
    
    # 计算SHAP值
    print("计算SHAP值...")
    shap_values = explainer.shap_values(features)
    
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
    
    # 保存绘图数据到JSON文件
    plot_data = {
        'shap_values': shap_values.tolist(),
        'feature_names': feature_names,
        'mean_abs_shap': mean_abs_shap.tolist()
    }
    
    with open(os.path.join(output_dir, 'shap_plot_data.json'), 'w') as f:
        json.dump(plot_data, f, indent=2)
    
    return sorted_features


def main():
    parser = argparse.ArgumentParser(description='VisNetV2 SHAP Analysis for Multiple Feature Types')
    
    # 数据和模型参数
    parser.add_argument('--model-path', type=str, 
                        default='./log/train_20251102-023702_dim48_layerH6_layerO6_batch64_lr0.0001_iter200/model.pt',
                        help='训练好的模型路径')
    parser.add_argument('--training-params-path', type=str,
                        default='./log/train_20251102-023702_dim48_layerH6_layerO6_batch64_lr0.0001_iter200/training_params.json',
                        help='训练参数文件路径')
    parser.add_argument('--data-path', type=str, default='./data/MMF-3/',
                        help='数据路径')
    parser.add_argument('--dataset-name', type=str, default='MMF_GNN_neg',
                        help='数据集名称')
    parser.add_argument('--output-dir', type=str, default='./results/shap',
                        help='SHAP结果输出目录')
    parser.add_argument('--sample-size', type=int, default=500,
                        help='用于SHAP分析的样本数量')
    parser.add_argument('--background-samples', type=int, default=100,
                        help='用于SHAP背景的样本数量')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='SHAP分析的批处理大小')
    parser.add_argument('--filter-by-tag', type=str, default=None,
                        help='根据标签过滤数据，如 Aromatic, Nitrogen_containing 等')
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
    
    # 获取标准化信息
    standardization_info = getattr(model, 'standardization_info', {})
    feature_level = getattr(model, 'feature_level', 'graph_physchem')
    physchem_mask = getattr(model, 'physchem_mask', None)
    toxicity_mask = getattr(model, 'toxicity_mask', None)
    chromato_mask = getattr(model, 'chromato_mask', None)
    
    print(f"模型使用的特征级别: {feature_level}")
    
    # 检查模型是否使用了特征标准化
    if len(standardization_info) > 0:
        print("检测到模型使用了特征标准化")
        print("物理化学特征标准化参数:")
        physchem_info = standardization_info.get('physchem', {})
        if physchem_info:
            print(f"  均值: {physchem_info.get('mean', [])}")
            print(f"  标准差: {physchem_info.get('std', [])}")
            
        if feature_level in ['graph_physchem_toxicity', 'all']:
            print("毒性特征标准化参数:")
            toxicity_info = standardization_info.get('toxicity', {})
            if toxicity_info:
                print(f"  均值: {toxicity_info.get('mean', [])}")
                print(f"  标准差: {toxicity_info.get('std', [])}")
                
        if feature_level == 'all':
            print("色谱特征标准化参数:")
            chromato_info = standardization_info.get('chromato', {})
            if chromato_info:
                print(f"  均值: {chromato_info.get('mean', [])}")
                print(f"  标准差: {chromato_info.get('std', [])}")
    else:
        print("模型未使用特征标准化")
        
    # 显示特征掩码信息
    if physchem_mask is not None:
        print(f"物理化学特征掩码: {physchem_mask}")
    if toxicity_mask is not None:
        print(f"毒性特征掩码: {toxicity_mask}")
    if chromato_mask is not None:
        print(f"色谱特征掩码: {chromato_mask}")
        
    # 准备数据
    print("准备数据...")
    
    # 模拟训练参数以匹配数据加载需求
    class Args:
        model = 'visnet_v2'
        visnet_v2_feature_level = feature_level
        debug_size = None
    
    args_obj = Args()
    
    # 加载数据集
    dataset_train, dataset_dev, dataset_test, _, _, _ = load_datasets(
        args_obj, args.data_path, args.dataset_name, max_data=None)
    
    # 如果指定了标签过滤，则根据标签过滤数据集
    if args.filter_by_tag:
        # 确定使用哪个标签文件
        if 'neg' in args.dataset_name:
            tag_file_path = os.path.join(args.data_path, 'tags', 'neg_compound_tags.csv')
        else:
            tag_file_path = os.path.join(args.data_path, 'tags', 'pos_compound_tags.csv')
        
        # 读取标签文件
        import pandas as pd
        tag_df = pd.read_csv(tag_file_path)
        
        # 过滤包含指定标签的化合物
        filtered_smiles = set()
        for _, row in tag_df.iterrows():
            smiles = row['SMILES']
            tags = row['Tags']
            if args.filter_by_tag in tags.split(','):
                filtered_smiles.add(smiles)
        
        print(f"根据标签 '{args.filter_by_tag}' 过滤数据，共找到 {len(filtered_smiles)} 个化合物")
        
        # 过滤数据集
        def filter_dataset(dataset, name):
            filtered_dataset = []
            for data in dataset:
                # 数据的第一个元素是SMILES字符串
                if data[0] in filtered_smiles:
                    filtered_dataset.append(data)
            print(f"数据集 {name} 过滤前: {len(dataset)} 个样本，过滤后: {len(filtered_dataset)} 个样本")
            return filtered_dataset
        
        dataset_train = filter_dataset(dataset_train, "train")
        dataset_dev = filter_dataset(dataset_dev, "dev")
        dataset_test = filter_dataset(dataset_test, "test")
    
    # VisNet系列模型需要特殊的预处理
    train_preprocessed_data, dev_preprocessed_data, test_preprocessed_data, \
    dataset_train_filtered, dataset_dev_filtered, dataset_test_filtered = preprocess_visnet_data(
        args_obj, dataset_train, dataset_dev, dataset_test, args.dataset_name, "visnet_train_" + args.dataset_name, "visnet_test_" + args.dataset_name)
    
    # 准备多种特征数据
    if args.use_combined_data:
        print("使用dev和test合并的数据集")
        features, smiles_list, feature_names = prepare_combined_multi_features_data(
            dataset_dev_filtered, dataset_test_filtered, dev_preprocessed_data, test_preprocessed_data, 
            feature_level, physchem_mask, toxicity_mask, chromato_mask, args.sample_size)
    elif args.use_train_data:
        print("使用train数据集")
        features, smiles_list, feature_names = prepare_multi_features_data(
            dataset_train_filtered, train_preprocessed_data, feature_level, 
            physchem_mask, toxicity_mask, chromato_mask, args.sample_size)
    else:
        print("使用test数据集")
        features, smiles_list, feature_names = prepare_multi_features_data(
            dataset_test_filtered, test_preprocessed_data, feature_level, 
            physchem_mask, toxicity_mask, chromato_mask, args.sample_size)
    
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
        model, features, reference_graph_data, device, args.background_samples, args.batch_size)
    
    # 分析特征重要性
    print("分析特征重要性...")
    feature_importance = analyze_feature_importance(shap_values, feature_names, output_dir_with_timestamp)
    
    # 使用自定义可视化函数
    print("生成可视化结果...")
    custom_visualize_shap_results(shap_values, features, feature_names, output_dir_with_timestamp)
    
    # 记录运行时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 保存运行信息
    runtime_info = {
        'start_time': datetime.fromtimestamp(start_time).isoformat(),
        'end_time': datetime.fromtimestamp(end_time).isoformat(),
        'elapsed_time_seconds': elapsed_time,
        'samples_used': len(features),
        'background_samples': args.background_samples,
        'batch_size': args.batch_size,
        'use_combined_data': args.use_combined_data,
        'feature_level': feature_level,
        'physchem_mask': physchem_mask,
        'toxicity_mask': toxicity_mask,
        'chromato_mask': chromato_mask,
        'physchem_dim': model.physchem_dim,
        'toxicity_dim': model.toxicity_dim,
        'chromato_dim': model.chromato_dim
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