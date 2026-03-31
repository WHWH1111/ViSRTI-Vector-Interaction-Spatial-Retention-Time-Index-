#!/usr/bin/env python3
"""
测试用。

VisNet V2预测脚本
专门处理特征掩码和标准化参数

针对大量数据预处理独立出来的版本。
"""

import os
from statistics import mean
import sys
import test
import torch
import numpy as np
import pandas as pd
from rdkit import Chem
import argparse
import json
import datetime
import csv
from datetime import datetime as dt
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 将项目根目录添加到Python路径中
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入VisNet V2模型
from models.visnet_v2 import VisNetV2
# 导入原子类型映射
from utils.atom_types import ATOM_TYPES
# 导入分子处理工具
from utils.molecule import MoleculeCache
# 导入Tester类
from core.trainer_tester import Tester
# 导入数据预处理工具
import preprocess as pp
# 导入数据预处理模块
from core.data_preprocessor import load_additional_features, filter_failed_molecules

# 物化特征名称到索引的映射
PHYSICOCHEMICAL_FEATURE_MAP = {
    'Monoiso_Mass': 0,
    'logKow/Exp_logKow': 1,
    'alogp/xlogp': 2,
    'Koc_predicted': 3
}

# 毒性特征名称到索引的映射
TOXICITY_FEATURE_MAP = {
    'Tetrahymena_pyriformis_toxicity': 0,
    'Daphnia_toxicity': 1,
    'Algae_toxicity': 2,
    'Pimephales_promelas_toxicity': 3
}

# 色谱特征名称到索引的映射
CHROMATO_FEATURE_MAP = {
    'Prob. +ESI': 0,
    'Prob. -ESI': 1
}

def load_model(model_path, params_path, device):
    """加载训练好的VisNet V2模型"""
    # 读取训练参数
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    # 确保是VisNet V2模型
    if params.get('model_type') != 'visnet_v2':
        raise ValueError("This script only supports visnet_v2 models")
    
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
        dropout_rate=params.get('visnet_v2_dropout_rate', 0.0),
        use_attention=params.get('visnet_v2_use_attention', False),
        use_gating=params.get('visnet_v2_use_gating', False),
        feature_level=params.get('visnet_v2_feature_level', 'all'),
        physchem_feature_mask=physchem_mask,
        toxicity_feature_mask=toxicity_mask,
        chromato_feature_mask=chromato_mask
    ).to(device)
    
    # 为了正确构建fusion_net，我们需要进行一次前向传播
    # 创建假的输入数据以触发fusion_net的构建
    feature_level = params.get('visnet_v2_feature_level', 'all')
    
    # 创建假的原子数据
    z = torch.zeros(2, dtype=torch.long).to(device)  # 原子类型
    pos = torch.zeros(2, 3).to(device)  # 原子坐标
    batch = torch.zeros(2, dtype=torch.long).to(device)  # 批次索引
    
    # 根据特征级别创建假的特征数据
    physchem_features = None
    toxicity_features = None
    chromato_features = None
    
    if feature_level in ['graph_physchem', 'graph_physchem_toxicity', 'all']:
        physchem_dim = params.get('visnet_v2_physchem_feature_dim', 4)
        # 应用掩码计算实际维度
        if physchem_mask is not None:
            physchem_dim = sum(physchem_mask)
        physchem_features = torch.zeros(1, physchem_dim).to(device)
        
    if feature_level in ['graph_physchem_toxicity', 'all']:
        toxicity_dim = params.get('visnet_v2_toxicity_feature_dim', 4)
        # 应用掩码计算实际维度
        if toxicity_mask is not None:
            toxicity_dim = sum(toxicity_mask)
        toxicity_features = torch.zeros(1, toxicity_dim).to(device)
        
    if feature_level == 'all':
        chromato_dim = params.get('visnet_v2_chromato_feature_dim', 2)
        # 应用掩码计算实际维度
        if chromato_mask is not None:
            chromato_dim = sum(chromato_mask)
        chromato_features = torch.zeros(1, chromato_dim).to(device)
    
    # 执行一次前向传播以构建完整的模型结构（包括fusion_net）
    with torch.no_grad():
        model.eval()
        try:
            model(z, pos, batch, physchem_features, toxicity_features, chromato_features)
        except Exception as e:
            # 忽略前向传播中的错误，我们只是想触发模型结构的构建
            pass
    
    # 现在加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()
    
    # 确保BatchNorm层处于评估状态
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            module.eval()
            # 确保使用训练时的统计信息
            module.track_running_stats = True
    
    return model, params


def standardize_features(features, means, stds):
    """使用给定的均值和标准差对特征进行标准化"""
    if features is None:
        return None
    
    # 标准化特征
    standardized_features = (features - means) / stds
    return standardized_features

def extract_features_from_row(row, feature_level):
    """从数据行中提取特征"""
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
        if not pd.isna(exp_logkow):
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
        if not pd.isna(xlogp):
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
    
    return physchem_features, toxicity_features, chromato_features

def calculate_metrics(predictions):
    """计算评估指标"""
    # 提取预测值和真实值
    y_true = np.array([p[2] for p in predictions])  # Actual values (已标准化)
    y_pred = np.array([p[1] for p in predictions])  # Predicted values (已标准化)
    
    # 检查并处理NaN值
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.all(valid_mask):
        print(f"Warning: Found {np.sum(~valid_mask)} NaN or infinite values in predictions. Removing them for metric calculation.")
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
    
    # 如果没有有效数据，返回空指标
    if len(y_true) == 0 or len(y_pred) == 0:
        print("Error: No valid data points for metric calculation.")
        return {
            'mae': float('nan'),
            'rmse': float('nan'),
            'r2': float('nan'),
            'mre': float('nan'),
            'valid_points': 0
        }
    
    # 计算各种评估指标（在标准化空间中）
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # 计算平均相对误差
    epsilon = 1e-8  # 避免除零错误
    mre = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mre': mre,  # Mean Relative Error (百分比)
        'valid_points': len(y_true)
    }

def predict_dataset(model, dataset, device, batch_size=64, params=None):
    """对整个数据集进行预测"""
    # 确保模型处于评估模式
    model.eval()
    
    # 确保BatchNorm层处于评估状态
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            module.eval()
            # 确保使用训练时的统计信息
            module.track_running_stats = True
    
    # 创建Tester实例，使用与训练时一致的缓存命名
    cache_name = f"visnet_predict_v2_{params.get('dataname', 'default')}"
    tester = Tester(model, visnet=True, batch_test=batch_size, device=device, name="predict-tester", MoleculeCacheName=cache_name)
    
    # 构建预处理数据字典，保持与训练时一致的格式
    preprocessed_data = {}
    for item in dataset:
        smiles = item['smiles']
        preprocessed_data[smiles] = item
        
    # 设置预处理数据（参考train.py中的调用方式）
    tester.set_preprocessed_data(preprocessed_data)
    
    # 设置属性标准化参数，与训练时保持一致
    if 'property_mean' in params and 'property_std' in params:
        tester.set_standardization_params(params['property_mean'], params['property_std'])
    
    # 使用Tester进行预测
    MAE, MSE, R2, PCC, predictions = tester.test_regressor(dataset)
    
    # 解析预测结果
    pred_lines = predictions.strip().split('\n')
    # print(pred_lines)
    parsed_predictions = []
    for line in pred_lines:
        parts = line.split('\t')
        if len(parts) == 3:
            smiles, true_value, pred_value = parts
            # 检查是否有NaN值
            if true_value.lower() != 'nan' and pred_value.lower() != 'nan':
                try:
                    parsed_predictions.append((smiles, float(pred_value), float(true_value)))
                except ValueError:
                    # 如果转换失败，跳过该行
                    print(f"Warning: Skipping invalid prediction for {smiles} (true: {true_value}, pred: {pred_value})")
                    continue
            else:
                print(f"Warning: Skipping NaN prediction for {smiles}")

    
    return parsed_predictions


def load_and_process_dataset_with_features(input_file, params, dataset_type=None, max_data=None):
    """加载和处理带额外特征的数据集，保持与训练时一致的处理方式"""
    # 获取数据集名称
    dataname = params.get('dataname', os.path.basename(input_file).replace('.csv', ''))
    
    # 确定目标列名
    target_column = 'Pred_RTI_Positive_ESI' if 'pos' in dataname.lower() else 'Pred_RTI_Negative_ESI'
    print('🐘 target_column: ', target_column)
    
    # 根据模型类型加载数据集
    feature_level = params.get('visnet_v2_feature_level', 'all')
    
    # 根据数据集类型确定要加载的数据文件
    if dataset_type == "train":
        dataset_file = f"data/MMF-3/{dataname}/{dataname}_train_set.txt"
        cache_name = f"visnet_train_v2_{dataname}"  # 修改缓存名称
    elif dataset_type == "dev":
        dataset_file = f"data/MMF-3/{dataname}/{dataname}_dev_set.txt"
        cache_name = f"visnet_dev_v2_{dataname}"   # 修改缓存名称
    elif dataset_type == "test":
        dataset_file = f"data/MMF-3/{dataname}/{dataname}_test_set.txt"
        cache_name = f"visnet_test_v2_{dataname}"  # 修改缓存名称
    else:  # 当dataset_type未指定或者不是预设值时，直接从CSV文件中读取
        dataset_file = None  # 不使用特定的数据集文件
        # 从输入文件路径中提取文件名作为标签
        input_filename = os.path.splitext(os.path.basename(input_file))[0]
        cache_name = f"visnet_predict_v2_{input_filename}"  # 使用预测缓存名称
    
    if dataset_file:
        print(f"Loading {dataset_type} dataset from {dataset_file}")
        
        # 读取训练/测试集文件中的SMILES
        smiles_list = []
        target_values = []
        with open(dataset_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    smiles_list.append(parts[0])
                    target_values.append(float(parts[1]))
    else:
        # 如果没有指定特定的数据集类型，则从CSV文件中加载所有数据
        print(f"Loading full dataset from {input_file}")
        df = pd.read_csv(input_file)
        smiles_list = df['SMILES'].tolist()
        # 尝试获取目标值列，如果没有则填充为0
        if target_column in df.columns:
            target_values = df[target_column].tolist()
        else:
            target_values = [0.0] * len(smiles_list)
            print(f"Warning: Target column '{target_column}' not found in CSV file")
    
    # 如果指定了最大数据量，则只处理这部分数据
    if max_data is not None:
        smiles_list = smiles_list[:max_data]
        target_values = target_values[:max_data]
        print(f"Processing only first {max_data} samples for debugging")
    
    train_cache = MoleculeCache(cache_name)
    types = ATOM_TYPES
    
    # 初始化计数器变量
    successful_count = 0
    failed_count = 0
    skipped_count = 0
    
    preprocessed_data = {}
    
    # 使用多线程处理SMILES列表
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Lock
    import multiprocessing
    
    # 创建线程锁以确保线程安全
    cache_lock = Lock()
    counters_lock = Lock()
    
    def process_single_smiles(smiles):
        nonlocal successful_count, failed_count, skipped_count
        # 首先检查是否之前已经标记为失败
        with cache_lock:
            if smiles in train_cache.failed_smiles:
                with counters_lock:
                    failed_count += 1
                    skipped_count += 1
                return None, "skipped"
        
        # 检查是否已经在缓存中（包括成功和失败的情况）
        with cache_lock:
            cached_result = train_cache.get(smiles, types)
        if cached_result is not None:
            # 成功缓存命中
            x, z, pos, edge_index, edge_attr = cached_result
            if (x is not None and z is not None and pos is not None and 
                edge_index is not None and edge_attr is not None):
                with counters_lock:
                    successful_count += 1
                return (smiles, {
                    'x': x,
                    'z': z,
                    'pos': pos,
                    'edge_index': edge_index,
                    'edge_attr': edge_attr
                }), "success"
            else:
                with counters_lock:
                    failed_count += 1
                return None, "failed"
        
        # 缓存未命中，计算分子图结构
        with cache_lock:
            result = train_cache.process_smiles(smiles, types)
        if result is not None and len(result) == 5 and result[0] is not None:
            x, z, pos, edge_index, edge_attr = result
            with counters_lock:
                successful_count += 1
            return (smiles, {
                'x': x,
                'z': z,
                'pos': pos,
                'edge_index': edge_index,
                'edge_attr': edge_attr
            }), "success"
        else:
            with counters_lock:
                failed_count += 1
            return None, "failed"
    
    # 获取CPU核心数，设置线程数为10
    max_workers = min(10, multiprocessing.cpu_count())
    print(f"使用 {max_workers} 个线程进行并行处理")
    
    # 并行处理SMILES
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_smiles = {executor.submit(process_single_smiles, smiles): smiles for smiles in smiles_list}
        
        # 处理完成的任务
        pbar = tqdm(as_completed(future_to_smiles), total=len(smiles_list), desc=f"Preprocessing {dataset_type} data")
        for future in pbar:
            result, status = future.result()
            if result is not None:
                smiles, data = result
                preprocessed_data[smiles] = data
            
            # 更新进度条描述信息
            with counters_lock:
                pbar.set_postfix({
                    'Success': successful_count,
                    'Failed': failed_count,
                    'Skipped': skipped_count
                })
    
    test_cache = MoleculeCache(cache_name)
    print(test_cache.get_data_size())
    
    return
    
    # 提取额外特征，保持与训练时一致的方式
    physchem_features_dict, toxicity_features_dict, chromato_features_dict, _ = load_additional_features(
        smiles_list, dataname, standardize=False, input_file=input_file)  # 先不标准化，下面使用 train 阶段中的参数来标准化
    
    # 应用训练时的标准化参数
    standardization_info = params.get('standardization_info', {})
    if standardization_info:
        print("Applying training standardization parameters...")
        physchem_means = np.array(standardization_info.get('physchem', {}).get('mean', [0]*4))
        physchem_stds = np.array(standardization_info.get('physchem', {}).get('std', [1]*4))
        toxicity_means = np.array(standardization_info.get('toxicity', {}).get('mean', [0]*4))
        toxicity_stds = np.array(standardization_info.get('toxicity', {}).get('std', [1]*4))
        chromato_means = np.array(standardization_info.get('chromato', {}).get('mean', [0]*2))
        chromato_stds = np.array(standardization_info.get('chromato', {}).get('std', [1]*2))
        
        # 对特征进行标准化
        for smiles in physchem_features_dict:
            if physchem_features_dict[smiles] is not None:
                physchem_features_dict[smiles] = ((physchem_features_dict[smiles] - physchem_means) / physchem_stds).astype(np.float32)
            if toxicity_features_dict[smiles] is not None:
                toxicity_features_dict[smiles] = ((toxicity_features_dict[smiles] - toxicity_means) / toxicity_stds).astype(np.float32)
            if chromato_features_dict[smiles] is not None:
                chromato_features_dict[smiles] = ((chromato_features_dict[smiles] - chromato_means) / chromato_stds).astype(np.float32)
    
    # 构建最终数据集，保持与训练时一致的格式
    additional_dataset = []
    property_mean = params.get('property_mean', 0)
    property_std = params.get('property_std', 1)
    
    # 读取CSV文件以获取额外特征
    df = pd.read_csv(input_file)
    smiles_to_row = {row['SMILES']: row for _, row in df.iterrows()}
    
    for i, smiles in enumerate(smiles_list):  # 修复：使用smiles_list而不是preprocessed_data.keys()
        # 检查SMILES是否在预处理数据中
        if smiles not in preprocessed_data:
            continue
            
        # 获取预处理的分子数据
        data = preprocessed_data[smiles].copy()
        data['smiles'] = smiles
        
        # 标准化属性值（与训练时保持一致）
        standardized_property_value = (target_values[i] - property_mean) / property_std
        data['property_value'] = standardized_property_value
        
        # 添加额外特征
        if smiles in smiles_to_row:
            row = smiles_to_row[smiles]
            data['physchem_features'] = physchem_features_dict.get(smiles, None)
            data['toxicity_features'] = toxicity_features_dict.get(smiles, None)
            data['chromato_features'] = chromato_features_dict.get(smiles, None)
        else:
            data['physchem_features'] = physchem_features_dict.get(smiles, None)
            data['toxicity_features'] = toxicity_features_dict.get(smiles, None)
            data['chromato_features'] = chromato_features_dict.get(smiles, None)
        
        additional_dataset.append(data)
    
    # 根据特征级别过滤特征，与训练时保持一致
    for data_dict in additional_dataset:
        smiles = data_dict['smiles']
        if feature_level == 'graph':
            # 只保留图特征，将其他特征设置为None
            data_dict['physchem_features'] = None
            data_dict['toxicity_features'] = None
            data_dict['chromato_features'] = None
        elif feature_level == 'graph_physchem':
            # 保留图特征和物化特征，将其他特征设置为None
            data_dict['toxicity_features'] = None
            data_dict['chromato_features'] = None
        elif feature_level == 'graph_physchem_toxicity':
            # 保留图特征、物化特征和毒性特征，将色谱特征设置为None
            data_dict['chromato_features'] = None
        # 如果是'all'级别，则保留所有特征
    
    print(f"Preprocessed {dataset_type} dataset: {len(smiles_list)} -> {len(additional_dataset)}")
    return additional_dataset


def process_dataset_file(filepath, target_column, smiles_column, filter_column=None, filter_value=None):
    """处理数据集文件"""
    # 读取数据
    df = pd.read_csv(filepath, low_memory=False)
    print(f"Loaded {len(df)} entries from {os.path.basename(filepath)}")
    
    # 仅在明确指定filter_column时才进行过滤
    if filter_column and filter_value is not None and filter_column in df.columns:
        original_count = len(df)
        df = df[df[filter_column] == filter_value]
        filtered_count = len(df)
        print(f"Filtered data from {original_count} to {filtered_count} entries "
              f"based on {filter_column} >= '{filter_value}'")
    
    return df

def save_predictions(predictions, output_path):
    """保存预测结果到CSV文件"""
    pred_df = pd.DataFrame(predictions, columns=['SMILES', 'Predicted', 'Actual'])
    pred_df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")

def save_metrics(metrics, output_path):
    """保存评估指标到JSON文件"""
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to {output_path}")

def main():
    np.random.seed(1234)
    torch.manual_seed(1234)
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Predict using trained VisNet model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--params_path', type=str, required=True, help='Path to the training parameters JSON file')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save predictions and metrics')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for prediction')
    parser.add_argument('--dataset_type', type=str, default='test', choices=['train', 'dev', 'test'], 
                        help='Dataset type to predict (train/dev/test)')
    parser.add_argument('--filter_column', type=str, help='Column name to filter data')
    parser.add_argument('--filter_value', type=float, help='Value to filter data (>= filter_value)')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 加载训练参数
    with open(args.params_path, 'r') as f:
        params = json.load(f)
    
    print("Loaded training parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # 加载模型
    print("Loading model...")
    model, params = load_model(args.model_path, args.params_path, device)
    print("Model loaded successfully")
    
    # 处理数据集，使用指定的数据集类型
    print(f"Processing {args.dataset_type} dataset...")
    dataset = load_and_process_dataset_with_features(args.input_file, params, args.dataset_type)
    
    # 进行预测
    print("Making predictions...")
    predictions = predict_dataset(model, dataset, device, args.batch_size, params)
    
    # 获取标准化参数
    property_mean = params.get('property_mean', 0)
    property_std = params.get('property_std', 1)
    
    # 保存预测结果
    timestamp = dt.now().strftime("%Y%m%d-%H%M%S")
    output_subdir = os.path.join(args.output_dir, f"prediction_{timestamp}")
    os.makedirs(output_subdir, exist_ok=True)
    
    # 保存预测结果到CSV文件（反标准化后的值）
    predictions_file = os.path.join(output_subdir, f"{args.dataset_type}_{os.path.splitext(os.path.basename(args.input_file))[0]}_predictions.csv")
    with open(predictions_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['SMILES', 'Predicted_Value', 'True_Value'])
        for pred in predictions:
            # 反标准化预测值和真实值
            smiles, standardized_pred, standardized_true = pred
            pred_value = standardized_pred * property_std + property_mean
            true_value = standardized_true * property_std + property_mean
            writer.writerow([smiles, pred_value, true_value])
    
    print(f"Saved predictions to {predictions_file}")
    
    # 计算并保存评估指标（在标准化空间中计算）
    print("Calculating metrics...")
    metrics = calculate_metrics(predictions)
    
    print("Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    metrics_file = os.path.join(output_subdir, f"{args.dataset_type}_{os.path.splitext(os.path.basename(args.input_file))[0]}_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Saved metrics to {metrics_file}")
    
    print("Prediction completed successfully!")

if __name__ == '__main__':
    main()