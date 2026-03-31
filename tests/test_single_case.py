#!/usr/bin/env python3
"""
测试单个案例的训练和预测一致性
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import json

# 将项目根目录添加到Python路径中
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入必要的模块
from models.visnet_v2 import VisNetV2
from utils.atom_types import ATOM_TYPES
from utils.molecule import MoleculeCache

# 测试用的SMILES
TEST_SMILES = "CCN(CCOC(=O)c1ccccc1)c1ccccc1"
TEST_CORRECT_VALUE = 0.6858579

def load_model_and_params(model_path, params_path, device):
    """加载训练好的VisNet V2模型和参数"""
    # 读取训练参数
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    print("Training parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # 获取特征掩码参数
    physchem_mask = params.get('visnet_v2_physchem_mask', None)
    toxicity_mask = params.get('visnet_v2_toxicity_mask', None)
    chromato_mask = params.get('visnet_v2_chromato_mask', None)
    
    # 创建VisNetV2模型实例，支持特征掩码
    model = VisNetV2(
        node_feature_dim=params.get('visnet_v2_node_feature_dim', 64),
        physchem_feature_dim=params.get('visnet_v2_physchem_feature_dim', 4),
        toxicity_feature_dim=params.get('visnet_v2_toxicity_feature_dim', 4),
        chromato_feature_dim=params.get('visnet_v2_chromato_feature_dim', 2),
        graph_hidden_dim=params.get('visnet_v2_graph_hidden_dim', 512),
        physchem_hidden_dim=params.get('visnet_v2_physchem_hidden_dim', 128),
        toxicity_hidden_dim=params.get('visnet_v2_toxicity_hidden_dim', 64),
        chromato_hidden_dim=params.get('visnet_v2_chromato_hidden_dim', 32),
        fusion_hidden_dims=params.get('visnet_v2_fusion_hidden_dims', [512, 256, 128]),
        dropout_rate=params.get('dropout', 0.3),
        use_attention=params.get('visnet_v2_use_attention', False),
        use_gating=params.get('visnet_v2_use_gating', False),
        feature_level=params.get('visnet_v2_feature_level', 'all'),
        physchem_feature_mask=physchem_mask,
        toxicity_feature_mask=toxicity_mask,
        chromato_feature_mask=chromato_mask
    ).to(device)
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()  # 设置模型为评估模式，这很重要，可以避免BatchNorm层在单样本预测时出现问题
    
    # 关键：确保所有的BatchNorm层都设置为eval模式
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            module.eval()
    
    return model, params

def process_single_molecule(smiles):
    """处理单个分子"""
    cache = MoleculeCache("test_molecule.pt")
    
    try:
        result = cache.get(smiles, ATOM_TYPES)
        if result is None:
            result = cache.process_smiles(smiles, ATOM_TYPES)
            if result is not None:
                cache.put(smiles, ATOM_TYPES, result)
        
        if result is not None and len(result) == 5:
            x, z, pos, edge_index, edge_attr = result
            if z is not None and pos is not None:
                return z, pos
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
        return None, None
    
    return None, None

def extract_features_for_smiles(smiles, df, feature_level):
    """从数据框中提取特定SMILES的特征"""
    row = df[df['SMILES'] == smiles].iloc[0]
    
    physchem_features = None
    toxicity_features = None
    chromato_features = None
    
    # 物化特征
    if feature_level in ['graph_physchem', 'graph_physchem_toxicity', 'all']:
        physchem_data = []
        # 1. Monoiso_Mass
        mass_value = row.get('Monoiso_Mass', 0)
        try:
            physchem_data.append(float(mass_value))
        except (ValueError, TypeError):
            physchem_data.append(0.0)
        
        # 2. logKow/Exp_logKow (优先使用实验值)
        exp_logkow = row.get('Exp_logKow_EPISuite')
        logkow = row.get('logKow_EPISuite', 0)
        if not pd.isna(exp_logkow) and exp_logkow != '':
            try:
                physchem_data.append(float(exp_logkow))
            except (ValueError, TypeError):
                physchem_data.append(0.0)
        else:
            try:
                physchem_data.append(float(logkow))
            except (ValueError, TypeError):
                physchem_data.append(0.0)
        
        # 3. alogp/xlogp (优先使用xlogp)
        xlogp = row.get('xlogp_ChemSpider')
        alogp = row.get('alogp_ChemSpider', 0)
        if not pd.isna(xlogp) and xlogp != '':
            try:
                physchem_data.append(float(xlogp))
            except (ValueError, TypeError):
                physchem_data.append(0.0)
        else:
            try:
                physchem_data.append(float(alogp))
            except (ValueError, TypeError):
                physchem_data.append(0.0)
        
        # 4. Koc_predicted - 与训练时保持一致，使用完整的列名
        koc_value = row.get('Koc_predicted (L/kg)', 0)
        try:
            physchem_data.append(float(koc_value))
        except (ValueError, TypeError):
            physchem_data.append(0.0)
        
        physchem_features = np.array(physchem_data, dtype=np.float32)
        print(f"Extracted physchem features: {physchem_features}")
    
    # 毒性特征
    if feature_level in ['graph_physchem_toxicity', 'all']:
        toxicity_data = []
        try:
            toxicity_data.append(float(row.get('Tetrahymena_pyriformis_toxicity', 0)))
        except (ValueError, TypeError):
            toxicity_data.append(0.0)
            
        try:
            toxicity_data.append(float(row.get('Daphnia_toxicity', 0)))
        except (ValueError, TypeError):
            toxicity_data.append(0.0)
            
        try:
            toxicity_data.append(float(row.get('Algae_toxicity', 0)))
        except (ValueError, TypeError):
            toxicity_data.append(0.0)
            
        try:
            toxicity_data.append(float(row.get('Pimephales_promelas_toxicity', 0)))
        except (ValueError, TypeError):
            toxicity_data.append(0.0)
        
        toxicity_features = np.array(toxicity_data, dtype=np.float32)
        print(f"Extracted toxicity features: {toxicity_features}")
    
    # 色谱特征
    if feature_level == 'all':
        chromato_data = []
        try:
            chromato_data.append(float(row.get('Prob. +ESI', 0)))
        except (ValueError, TypeError):
            chromato_data.append(0.0)
            
        try:
            chromato_data.append(float(row.get('Prob. -ESI', 0)))
        except (ValueError, TypeError):
            chromato_data.append(0.0)
        
        chromato_features = np.array(chromato_data, dtype=np.float32)
        print(f"Extracted chromato features: {chromato_features}")
    
    return physchem_features, toxicity_features, chromato_features

def standardize_features(features, means, stds):
    """使用给定的均值和标准差对特征进行标准化"""
    if features is None:
        return None
    
    # 标准化特征
    standardized_features = (features - means) / stds
    return standardized_features

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 模型和参数路径
    dir_name = "train_20251117-020257_dim48_layerH6_layerO6_batch64_lr0.0001_iter1"
    model_path = f"./log/{dir_name}/model.pt"
    params_path = f"./log/{dir_name}/training_params.json"
    
    # 加载模型和参数
    print("Loading model...")
    model, params = load_model_and_params(model_path, params_path, device)
    print("Model loaded successfully")
    
    # 处理分子
    print(f"Processing molecule: {TEST_SMILES}")
    z, pos = process_single_molecule(TEST_SMILES)
    
    if z is None or pos is None:
        print("Failed to process molecule")
        return
    
    # 确保分子数据在正确的设备上并具有正确的维度
    z = z.to(device)  # 原子序数不需要批次维度
    pos = pos.to(device)  # 原子位置不需要批次维度
    batch_tensor = torch.zeros(z.shape[0], dtype=torch.long).to(device)  # 为每个原子创建批次索引
    
    # 读取数据文件以获取特征
    df = pd.read_csv("./data/MMF-3/MMF_GNN_pos.csv")
    
    # 获取特征级别
    feature_level = params.get('visnet_v2_feature_level', 'all')
    
    # 提取特征
    physchem_features, toxicity_features, chromato_features = extract_features_for_smiles(TEST_SMILES, df, feature_level)
    
    # 获取标准化参数
    property_mean = params.get('property_mean', 0)
    property_std = params.get('property_std', 1)
    
    # 获取特征标准化参数
    standardization_info = params.get('standardization_info', {})
    physchem_means = standardization_info.get('physchem', {}).get('mean', [0]*4)
    physchem_stds = standardization_info.get('physchem', {}).get('std', [1]*4)
    toxicity_means = standardization_info.get('toxicity', {}).get('mean', [0]*4)
    toxicity_stds = standardization_info.get('toxicity', {}).get('std', [1]*4)
    chromato_means = standardization_info.get('chromato', {}).get('mean', [0]*2)
    chromato_stds = standardization_info.get('chromato', {}).get('std', [1]*2)
    
    # 获取特征掩码
    physchem_mask = params.get('visnet_v2_physchem_mask', None)
    toxicity_mask = params.get('visnet_v2_toxicity_mask', None)
    chromato_mask = params.get('visnet_v2_chromato_mask', None)
    
    print(f"Standardization info:")
    print(f"  Property mean: {property_mean}, std: {property_std}")
    print(f"  Physchem means: {physchem_means}")
    print(f"  Physchem stds: {physchem_stds}")
    print(f"  Toxicity means: {toxicity_means}")
    print(f"  Toxicity stds: {toxicity_stds}")
    print(f"  Physchem mask: {physchem_mask}")
    print(f"  Toxicity mask: {toxicity_mask}")
    
    # 标准化特征，但不应用掩码（让模型内部处理）
    if physchem_features is not None:
        physchem_features = standardize_features(physchem_features, physchem_means, physchem_stds)
        print(f"Standardized physchem features: {physchem_features}")
        physchem_features = torch.tensor(physchem_features, dtype=torch.float32).unsqueeze(0).to(device)
    
    if toxicity_features is not None:
        toxicity_features = standardize_features(toxicity_features, toxicity_means, toxicity_stds)
        print(f"Standardized toxicity features: {toxicity_features}")
        toxicity_features = torch.tensor(toxicity_features, dtype=torch.float32).unsqueeze(0).to(device)
    
    # 使用已创建的 batch_tensor，无需重新定义
    # batch_tensor 已经是长度为原子数、值全为0的张量，表示单个图
    
    # 进行预测
    print("Making prediction...")
    with torch.no_grad():
        predicted_value, _ = model(z, pos, batch_tensor, physchem_features, toxicity_features, chromato_features)
    
    # 反标准化预测值
    if isinstance(predicted_value, torch.Tensor):
        predicted_value = predicted_value.cpu().numpy()
    final_prediction = predicted_value * property_std + property_mean
    
    print(f"Final prediction: {final_prediction}")
    print(f"Correct value: {TEST_CORRECT_VALUE}")
    print(f"Difference: {abs(final_prediction - TEST_CORRECT_VALUE)}")
    
    # 检查训练时的预测值
    train_prediction = 0.38002682  # 从训练预测文件中获取
    print(f"Training prediction: {train_prediction}")
    print(f"Difference from training: {abs(final_prediction - train_prediction)}")

if __name__ == "__main__":
    main()