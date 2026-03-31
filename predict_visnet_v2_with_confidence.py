#!/usr/bin/env python3
"""
VisNet V2预测脚本（含置信度计算）- 直接MC-Dropout实现
专门处理特征掩码和标准化参数，使用蒙特卡洛Dropout方法计算预测置信度
绕过Tester类，直接实现MC-Dropout
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from rdkit import Chem
import argparse
import json
import datetime
import traceback
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
        toxicity_intermediate_dim=params.get('visnet_v2_toxicity_intermediate_dim', 64),
        chromato_hidden_dim=params.get('visnet_v2_chromato_hidden_dim', 32),
        fusion_hidden_dims=params.get('visnet_v2_fusion_hidden_dims', [512, 256, 128]),
        dropout_rate=params.get('visnet_v2_dropout_rate', 0.1),  # 使用训练时的dropout_rate
        use_attention=params.get('visnet_v2_use_attention', False),
        use_gating=params.get('visnet_v2_use_gating', False),
        feature_level=params.get('visnet_v2_feature_level', 'all'),
        physchem_feature_mask=physchem_mask,
        toxicity_feature_mask=toxicity_mask,
        chromato_feature_mask=chromato_mask
    ).to(device)
    
    # 构建fusion_net - 使用与训练时相同的方法
    fusion_input_dim = model.get_fusion_net_input_dim()
    model._build_fusion_net(fusion_input_dim)
    model._last_feature_dim = fusion_input_dim
    model.fusion_net = model.fusion_net.to(device)
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    
    # 初始设置为评估模式
    model.eval()
    
    # 确保BatchNorm层处于评估状态
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            module.eval()
            module.track_running_stats = True
    
    return model, params

def enable_mc_dropout(model):
    """启用MC-Dropout模式：启用Dropout但保持BatchNorm在评估模式"""
    model.train()  # 启用Dropout
    
    # 但保持BatchNorm在评估模式
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            module.eval()
            module.track_running_stats = True

def prepare_mc_dropout_batch(batch_data, device):
    """准备MC-Dropout批次数据"""
    z_list, pos_list, batch_indices_list, smiles_list = [], [], [], []
    physchem_list, toxicity_list, chromato_list = [], [], []
    
    batch_idx = 0
    
    for data in batch_data:
        smiles = data.get('smiles')
        if not smiles:
            continue
            
        # 提取图数据
        z = data.get('z')
        pos = data.get('pos')
        if z is None or pos is None:
            continue
        
        # 转换为张量
        z_tensor = torch.tensor(z, device=device) if not isinstance(z, torch.Tensor) else z.to(device)
        pos_tensor = torch.tensor(pos, device=device) if not isinstance(pos, torch.Tensor) else pos.to(device)
        
        z_list.append(z_tensor)
        pos_list.append(pos_tensor)
        batch_indices_list.append(torch.full((z_tensor.shape[0],), batch_idx, dtype=torch.long, device=device))
        smiles_list.append(smiles)
        
        # 提取额外特征
        physchem = data.get('physchem_features')
        toxicity = data.get('toxicity_features') 
        chromato = data.get('chromato_features')
        
        if physchem is not None:
            physchem_tensor = torch.tensor(physchem, device=device, dtype=torch.float32)
            physchem_list.append(physchem_tensor.unsqueeze(0))
        if toxicity is not None:
            toxicity_tensor = torch.tensor(toxicity, device=device, dtype=torch.float32)
            toxicity_list.append(toxicity_tensor.unsqueeze(0))
        if chromato is not None:
            chromato_tensor = torch.tensor(chromato, device=device, dtype=torch.float32)
            chromato_list.append(chromato_tensor.unsqueeze(0))
        
        batch_idx += 1
    
    if len(z_list) == 0:
        return None
    
    # 合并批次
    try:
        z_batch = torch.cat(z_list, dim=0)
        pos_batch = torch.cat(pos_list, dim=0)
        batch_indices = torch.cat(batch_indices_list, dim=0)
        
        # 处理额外特征
        physchem_features = torch.cat(physchem_list, dim=0) if physchem_list else None
        toxicity_features = torch.cat(toxicity_list, dim=0) if toxicity_list else None
        chromato_features = torch.cat(chromato_list, dim=0) if chromato_list else None
        
        return z_batch, pos_batch, batch_indices, smiles_list, (physchem_features, toxicity_features, chromato_features)
    
    except Exception as e:
        print(f"Error preparing batch: {e}")
        return None

def mc_dropout_predict(model, dataset, device, batch_size=64, n_iterations=100):
    """MC-Dropout预测主函数"""
    # 启用MC-Dropout模式
    enable_mc_dropout(model)
    
    all_predictions = {}
    
    print(f"Performing {n_iterations} MC-Dropout iterations...")
    
    for iteration in tqdm(range(n_iterations), desc="MC-Dropout sampling"):
        # 批次处理
        for i in range(0, len(dataset), batch_size):
            batch_data = dataset[i:i+batch_size]
            
            # 准备批次数据
            batch_results = prepare_mc_dropout_batch(batch_data, device)
            if batch_results is None:
                continue
                
            z_batch, pos_batch, batch_indices, smiles_list, additional_features = batch_results
            
            with torch.no_grad():  # 不需要梯度，但Dropout仍然工作
                try:
                    # 前向传播
                    physchem_features, toxicity_features, chromato_features = additional_features
                    pred = model(z_batch, pos_batch, batch_indices, physchem_features, toxicity_features, chromato_features)
                    
                    if isinstance(pred, tuple):
                        pred = pred[0]  # VisNetV2返回(pred, features)
                    
                    pred_values = pred.cpu().numpy().flatten()
                    
                    # 收集预测结果
                    for idx, smiles in enumerate(smiles_list):
                        if idx >= len(pred_values):  # 确保不越界
                            continue
                            
                        if smiles not in all_predictions:
                            all_predictions[smiles] = {
                                'predictions': [],
                                'true_value': dataset[i + idx].get('property_value', 0)
                            }
                        all_predictions[smiles]['predictions'].append(pred_values[idx])
                        
                except Exception as e:
                    print(f"Error in forward pass for batch {i}: {e}")
                    continue
    
    return all_predictions

def calculate_confidence_scores(all_predictions, property_mean, property_std):
    """计算置信度分数 - 改进版本"""
    confidence_results = []
    
    for smiles, data in all_predictions.items():
        predictions = np.array(data['predictions'])
        if len(predictions) == 0:
            continue
            
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        # 反标准化预测值
        mean_pred_denorm = mean_pred * property_std + property_mean
        true_value_denorm = data['true_value'] * property_std + property_mean
        
        # 改进的置信度计算
        if std_pred == 0:
            confidence = 1.0  # 所有预测一致，高置信度
        else:
            # 方法1: 基于标准差的改进置信度
            # 将标准差缩放到与误差相近的范围
            normalized_std = std_pred * property_std
            confidence_std = 1.0 / (1.0 + normalized_std / 50.0)  # 50是经验值
            
            # 方法2: 基于预测范围的置信度
            pred_range = np.max(predictions) - np.min(predictions)
            # print(pred_range, property_std)
            confidence_range = 1.0 / (1.0 + pred_range * property_std / 100.0)
            
            # 方法3: 组合置信度
            confidence = 0.6 * confidence_std + 0.4 * confidence_range
        
        confidence_results.append({
            'smiles': smiles,
            'predicted': mean_pred_denorm,  # 反标准化的预测值
            'true_value': true_value_denorm,  # 反标准化的真实值
            'std': std_pred * property_std,  # 反标准化的标准差
            'confidence': confidence,
            'min_pred': np.min(predictions) * property_std + property_mean,
            'max_pred': np.max(predictions) * property_std + property_mean,
            'n_samples': len(predictions),
            'pred_range': (np.max(predictions) - np.min(predictions)) * property_std  # 添加预测范围
        })
    
    return confidence_results

def predict_with_confidence(model, dataset, device, batch_size=64, params=None, n_iterations=100):
    """对整个数据集进行预测并计算置信度 - 直接MC-Dropout实现"""
    
    # 进行MC-Dropout预测
    all_predictions = mc_dropout_predict(
        model, dataset, device, batch_size, n_iterations
    )
    
    # 获取标准化参数
    property_mean = params.get('property_mean', 0)
    property_std = params.get('property_std', 1)
    
    # 计算置信度分数
    confidence_results = calculate_confidence_scores(all_predictions, property_mean, property_std)
    
    return confidence_results

def load_and_process_dataset_with_features(input_file, params, dataset_type=None, max_data=None, pass_items=None):
    """加载和处理带额外特征的数据集，保持与训练时一致的处理方式"""
    # 获取数据集名称
    dataname = params.get('dataname', os.path.basename(input_file).replace('.csv', ''))
    
    # 确定目标列名
    target_column = 'Pred_RTI_Positive_ESI' if 'pos' in dataname.lower() else 'Pred_RTI_Negative_ESI'
    print('Target column: ', target_column)
    
    # 根据模型类型加载数据集
    feature_level = params.get('visnet_v2_feature_level', 'all')
    
    # 根据数据集类型确定要加载的数据文件
    if dataset_type == "train":
        dataset_file = f"data/MMF-3/{dataname}/{dataname}_train_set.txt"
        cache_name = f"visnet_train_v2_{dataname}"
    elif dataset_type == "dev":
        dataset_file = f"data/MMF-3/{dataname}/{dataname}_dev_set.txt"
        cache_name = f"visnet_train_v2_{dataname}"
    elif dataset_type == "test":
        dataset_file = f"data/MMF-3/{dataname}/{dataname}_test_set.txt"
        cache_name = f"visnet_test_v2_{dataname}"
    else:
        dataset_file = None
        input_filename = os.path.splitext(os.path.basename(input_file))[0]
        cache_name = f"visnet_predict_v2_{input_filename}"
    
    if dataset_file:
        print(f"Loading {dataset_type} dataset from {dataset_file}")
        smiles_list = []
        target_values = []
        with open(dataset_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    smiles_list.append(parts[0])
                    target_values.append(float(parts[1]))
    else:
        print(f"Loading full dataset from {input_file}")
        df = pd.read_csv(input_file)
        smiles_list = df['SMILES'].tolist()
        if target_column in df.columns:
            target_values = df[target_column].tolist()
        else:
            target_values = [0.0] * len(smiles_list)
            print(f"Warning: Target column '{target_column}' not found in CSV file")
    
    # 处理数据切片
    if pass_items is not None and pass_items > 0:
        print(f"Skipping first {pass_items} items...")
        smiles_list = smiles_list[pass_items:]
        target_values = target_values[pass_items:]
    
    if max_data is not None:
        smiles_list = smiles_list[:(max_data - pass_items) if pass_items else max_data]
        target_values = target_values[:(max_data - pass_items) if pass_items else max_data]
        print(f"Processing only {(max_data - pass_items) if pass_items else max_data} samples for debugging")
    
    train_cache = MoleculeCache(cache_name)
    types = ATOM_TYPES
    
    # 预处理数据
    successful_count = 0
    failed_count = 0
    skipped_count = 0
    
    preprocessed_data = {}
    pbar = tqdm(smiles_list, desc=f"Preprocessing {dataset_type} data")
    for smiles in pbar:
        if smiles in train_cache.failed_smiles:
            failed_count += 1
            skipped_count += 1
            continue
            
        cached_result = train_cache.get(smiles, types)
        if cached_result is not None:
            x, z, pos, edge_index, edge_attr = cached_result
            if (x is not None and z is not None and pos is not None and 
                edge_index is not None and edge_attr is not None):
                preprocessed_data[smiles] = {
                    'x': x,
                    'z': z,
                    'pos': pos,
                    'edge_index': edge_index,
                    'edge_attr': edge_attr
                }
                successful_count += 1
            else:
                failed_count += 1
            continue
        
        failed_count += 1
        continue  # 跳过缓存未命中的处理
        
        pbar.set_postfix({
            'Success': successful_count,
            'Failed': failed_count,
            'Skipped': skipped_count
        })
    
    # 提取额外特征
    physchem_features_dict, toxicity_features_dict, chromato_features_dict, _ = load_additional_features(
        smiles_list, dataname, standardize=False, input_file=input_file)
    
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
        
        for smiles in physchem_features_dict:
            if physchem_features_dict[smiles] is not None:
                physchem_features_dict[smiles] = ((physchem_features_dict[smiles] - physchem_means) / physchem_stds).astype(np.float32)
            if toxicity_features_dict[smiles] is not None:
                toxicity_features_dict[smiles] = ((toxicity_features_dict[smiles] - toxicity_means) / toxicity_stds).astype(np.float32)
            if chromato_features_dict[smiles] is not None:
                chromato_features_dict[smiles] = ((chromato_features_dict[smiles] - chromato_means) / chromato_stds).astype(np.float32)
    
    # 构建最终数据集
    additional_dataset = []
    property_mean = params.get('property_mean', 0)
    property_std = params.get('property_std', 1)
    
    df = pd.read_csv(input_file)
    smiles_to_row = {row['SMILES']: row for _, row in df.iterrows()}
    
    for i, smiles in enumerate(smiles_list):
        if smiles not in preprocessed_data:
            continue
            
        data = preprocessed_data[smiles].copy()
        data['smiles'] = smiles
        
        # 标准化属性值
        standardized_property_value = (target_values[i] - property_mean) / property_std
        data['property_value'] = standardized_property_value
        
        # 添加额外特征
        if smiles in smiles_to_row:
            data['physchem_features'] = physchem_features_dict.get(smiles, None)
            data['toxicity_features'] = toxicity_features_dict.get(smiles, None)
            data['chromato_features'] = chromato_features_dict.get(smiles, None)
        else:
            data['physchem_features'] = physchem_features_dict.get(smiles, None)
            data['toxicity_features'] = toxicity_features_dict.get(smiles, None)
            data['chromato_features'] = chromato_features_dict.get(smiles, None)
        
        additional_dataset.append(data)
    
    # 根据特征级别过滤特征
    for data_dict in additional_dataset:
        if feature_level == 'graph':
            data_dict['physchem_features'] = None
            data_dict['toxicity_features'] = None
            data_dict['chromato_features'] = None
        elif feature_level == 'graph_physchem':
            data_dict['toxicity_features'] = None
            data_dict['chromato_features'] = None
        elif feature_level == 'graph_physchem_toxicity':
            data_dict['chromato_features'] = None
    
    print(f"Preprocessed {dataset_type} dataset: {len(smiles_list)} -> {len(additional_dataset)}")
    return additional_dataset

def calculate_metrics(predictions, property_mean, property_std):
    """计算评估指标"""
    smiles_list = [p[0] for p in predictions]
    y_true = np.array([p[2] for p in predictions])
    y_pred = np.array([p[1] for p in predictions])
    
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    invalid_count = np.sum(~valid_mask)
    if invalid_count > 0:
        print(f"Warning: Found {invalid_count} NaN or infinite values in predictions. Removing them for metric calculation.")
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
        smiles_list = [smiles_list[i] for i in range(len(smiles_list)) if valid_mask[i]]
    
    if len(y_true) == 0 or len(y_pred) == 0:
        print("Error: No valid data points for metric calculation.")
        return {
            'mae': float('nan'),
            'rmse': float('nan'),
            'r2': float('nan'),
            'mre': float('nan'),
            'valid_points': 0
        }
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    epsilon = 1e-8
    mre = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100
    
    return {
        'mae': mae,
        'mae_norm': mae / property_std,
        'mse': mse,
        'mse_norm': mse / (property_std ** 2),
        'rmse': rmse,
        'rmse_norm': rmse / property_std,
        'r2': r2,
        'mre': mre,
        'valid_points': len(y_true)
    }

def save_predictions_with_confidence(predictions, output_path, id_mapping=None):
    """保存带置信度的预测结果到CSV文件"""
    if id_mapping is not None and len(id_mapping) > 0:
        pred_data = []
        for pred in predictions:
            smiles = pred['smiles']
            compound_id = id_mapping.get(smiles, '')
            pred_data.append((
                compound_id, 
                smiles, 
                pred['predicted'], 
                pred['true_value'],
                pred['std'],
                pred['confidence'],
                pred['min_pred'],
                pred['max_pred'],
                pred['n_samples'],
                pred.get('pred_range', 0.0)  # 添加预测范围
            ))
        pred_df = pd.DataFrame(pred_data, columns=[
            'Norman_SusDat_ID', 'SMILES', 'Predicted', 'Actual', 'Std', 'Confidence', 'Min_Pred', 'Max_Pred', 'N_Samples', 'Pred_Range'
        ])
    else:
        pred_data = []
        for pred in predictions:
            pred_data.append((
                pred['smiles'],
                pred['predicted'],
                pred['true_value'],
                pred['std'],
                pred['confidence'],
                pred['min_pred'],
                pred['max_pred'],
                pred['n_samples'],
                pred.get('pred_range', 0.0)  # 添加预测范围
            ))
        pred_df = pd.DataFrame(pred_data, columns=[
            'SMILES', 'Predicted', 'Actual', 'Std', 'Confidence', 'Min_Pred', 'Max_Pred', 'N_Samples', 'Pred_Range'
        ])
    
    pred_df.to_csv(output_path, index=False)
    print(f"Saved predictions with confidence to {output_path}")

def save_metrics(metrics, output_path):
    """保存评估指标到JSON文件"""
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to {output_path}")

def debug_mc_dropout(model, dataset, device, n_samples=5):
    """调试MC-Dropout是否正常工作"""
    enable_mc_dropout(model)
    
    # 测试第一个样本
    test_sample = dataset[0]
    smiles = test_sample['smiles']
    
    print(f"\n=== MC-Dropout调试: {smiles} ===")
    
    predictions = []
    for i in range(n_samples):
        # 准备单个样本
        batch_results = prepare_mc_dropout_batch([test_sample], device)
        if batch_results is None:
            continue
            
        z_batch, pos_batch, batch_indices, _, additional_features = batch_results
        
        with torch.no_grad():
            physchem_features, toxicity_features, chromato_features = additional_features
            pred = model(z_batch, pos_batch, batch_indices, physchem_features, toxicity_features, chromato_features)
            
            if isinstance(pred, tuple):
                pred = pred[0]
            
            pred_value = pred.cpu().numpy().flatten()[0]
            predictions.append(pred_value)
            print(f"迭代 {i+1}: {pred_value:.3f}")
    
    if predictions:
        print(f"预测值范围: {min(predictions):.3f} - {max(predictions):.3f}")
        print(f"标准差: {np.std(predictions):.3f}")
        print(f"均值: {np.mean(predictions):.3f}")
    
    return predictions

def check_dropout_layers(model):
    """检查模型中的Dropout层"""
    dropout_count = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            print(f"Dropout层: {name}, rate={module.p}")
            dropout_count += 1
    print(f"总共找到 {dropout_count} 个Dropout层")
    return dropout_count > 0

def verify_model_mode(model):
    """验证模型模式设置"""
    print("\n=== 模型模式验证 ===")
    print(f"模型训练模式: {model.training}")
    
    dropout_modules = []
    batchnorm_modules = []
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            dropout_modules.append((name, module.training))
        elif isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            batchnorm_modules.append((name, module.training))
    
    print("Dropout层状态:")
    for name, training in dropout_modules:
        print(f"  {name}: {'训练模式' if training else '评估模式'}")
    
    print("BatchNorm层状态:")
    for name, training in batchnorm_modules:
        print(f"  {name}: {'训练模式' if training else '评估模式'}")

def main():
    np.random.seed(1234)
    torch.manual_seed(1234)
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Predict using trained VisNet model with MC-Dropout confidence calculation')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--params_path', type=str, required=True, help='Path to the training parameters JSON file')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save predictions and metrics')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for prediction')
    parser.add_argument('--dataset_type', type=str, choices=['train', 'dev', 'test'], 
                        help='Dataset type to predict (train/dev/test)')
    parser.add_argument('--filter_column', type=str, help='Column name to filter data')
    parser.add_argument('--filter_value', type=float, help='Value to filter data (>= filter_value)')
    parser.add_argument('--max_items', type=int, help='Maximum number of items to process (useful for testing)')
    parser.add_argument('--pass_items', type=int, help='Number of items to skip at the beginning of the dataset')
    parser.add_argument('--n_iterations', type=int, default=50, help='Number of iterations for MC-Dropout confidence calculation')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
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
    
    # 如果启用了调试模式，进行调试
    if args.debug:
        print("Running debug checks...")
        check_dropout_layers(model)
        verify_model_mode(model)
    
    # 处理数据集
    print(f"Processing {args.dataset_type} dataset...")
    dataset = load_and_process_dataset_with_features(args.input_file, params, args.dataset_type, args.max_items, args.pass_items)
    
    # 如果启用了调试模式，进行MC-Dropout调试
    if args.debug:
        debug_mc_dropout(model, dataset[:1], device, n_samples=10)
    
    # 进行MC-Dropout预测并计算置信度
    print("Making predictions with MC-Dropout confidence calculation...")
    predictions = predict_with_confidence(
        model, dataset, device, args.batch_size, params, args.n_iterations
    )
    
    # 获取标准化参数
    property_mean = params.get('property_mean', 0)
    property_std = params.get('property_std', 1)
    
    # 保存预测结果
    timestamp = dt.now().strftime("%Y%m%d-%H%M%S")
    output_subdir = os.path.join(args.output_dir, f"mc_dropout_prediction_{timestamp}")
    os.makedirs(output_subdir, exist_ok=True)
    
    # 检查输入文件并创建ID映射
    df_input = pd.read_csv(args.input_file)
    id_mapping = {}
    
    id_columns = ['Norman_SusDat_ID', 'Compound_ID', 'ID', 'Compound_Name']
    found_id_column = None
    
    for col in id_columns:
        if col in df_input.columns:
            found_id_column = col
            break
    
    if found_id_column:
        id_mapping = dict(zip(df_input['SMILES'], df_input[found_id_column]))
        print(f"Found ID column: {found_id_column}, mapping {len(id_mapping)} compounds")
    else:
        print("No recognized ID column found in input file")
    
    # 保存预测结果到CSV文件
    predictions_file = os.path.join(output_subdir, f"{args.dataset_type}_{os.path.splitext(os.path.basename(args.input_file))[0]}_mc_dropout_predictions.csv")
    save_predictions_with_confidence(predictions, predictions_file, id_mapping if id_mapping else None)
    
    print(f"Saved predictions to {predictions_file}")
    
    # 计算并保存评估指标
    print("Calculating metrics...")
    simple_predictions = [(p['smiles'], p['predicted'], p['true_value']) for p in predictions]
    metrics = calculate_metrics(simple_predictions, property_mean, property_std)
    
    print("Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    metrics_file = os.path.join(output_subdir, f"{args.dataset_type}_{os.path.splitext(os.path.basename(args.input_file))[0]}_metrics.json")
    save_metrics(metrics, metrics_file)
    
    print("MC-Dropout prediction with confidence calculation completed successfully!")

if __name__ == '__main__':
    main()