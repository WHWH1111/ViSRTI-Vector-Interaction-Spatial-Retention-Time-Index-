#!/usr/bin/env python3
"""
Predict script for GNN model to validate performance on MMF dataset subsets
"""

import os
import sys
import torch
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from rdkit import Chem
import argparse
import json
import datetime
from tqdm import tqdm

# 将项目根目录添加到Python路径中
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import preprocess as pp
from train import MolecularGraphNeuralNetwork
# 导入VisNet模型
from models.visnet import ViSNet
# 导入原子类型映射
from utils.atom_types import ATOM_TYPES
# 导入新创建的工具函数
from utils.evaluation_utils import calculate_evaluation_metrics, calculate_accuracy_metrics, generate_metrics_report, save_metrics_report
from utils.prediction_plot_utils import plot_error_distribution

# 初始化字典
atom_dict = defaultdict(lambda: len(atom_dict))
bond_dict = defaultdict(lambda: len(bond_dict))
fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
edge_dict = defaultdict(lambda: len(edge_dict))
radius = 1

# 添加全局变量来统计跳过的分子数量
skipped_molecules_count = 0

# 添加列表来记录跳过的分子及其原因
skipped_molecules_details = []

# 添加分子缓存
mol_cache = None

def load_model(model_path, params_path, device):
    """加载训练好的模型"""
    # 读取训练参数
    with open(params_path, 'r') as f:
        import json
        params = json.load(f)
    
    # 检查是否是VisNet模型
    if params.get('model_type') == 'visnet':
        # 创建VisNet模型实例
        model = ViSNet(
            hidden_channels=params.get('visnet_hidden_channels', 128),
            num_layers=params.get('visnet_num_layers', 6),
            num_heads=params.get('visnet_num_heads', 8),
            num_rbf=params.get('visnet_num_rbf', 32),
            cutoff=params.get('visnet_cutoff', 5.0),
        ).to(device)
    elif params.get('model_type') == 'visnet_v1' or params.get('visnet_mfe', False):
        from models.visnet import VisNetV1
        # 创建VisNetV1模型实例
        model = VisNetV1(
            node_feature_dim=params.get('visnet_mfe_node_feature_dim', 64),
            hidden_dims=params.get('visnet_mfe_hidden_dims', [128, 256]),
            output_dims=params.get('visnet_mfe_output_dims', [256, 128, 1]),
            dropout_rate=params.get('dropout', 0.5)  # 使用统一的dropout参数
        ).to(device)
    elif params.get('model_type') == 'visnet_v2' or params.get('visnet_v2', False):
        from models.visnet_v2 import VisNetV2
        # 创建VisNetV2模型实例
        model = VisNetV2(
            node_feature_dim=params.get('visnet_v2_node_feature_dim', 64),
            physchem_feature_dim=params.get('visnet_v2_physchem_feature_dim', 12),
            toxicity_feature_dim=params.get('visnet_v2_toxicity_feature_dim', 4),
            chromato_feature_dim=params.get('visnet_v2_chromato_feature_dim', 3),
            graph_hidden_dim=params.get('visnet_v2_graph_hidden_dim', 512),
            physchem_hidden_dim=params.get('visnet_v2_physchem_hidden_dim', 128),
            toxicity_hidden_dim=params.get('visnet_v2_toxicity_hidden_dim', 64),
            chromato_hidden_dim=params.get('visnet_v2_chromato_hidden_dim', 32),
            fusion_hidden_dims=params.get('visnet_v2_fusion_hidden_dims', [512, 256, 128]),
            dropout_rate=params.get('dropout', 0.3)  # 使用统一的dropout参数
        ).to(device)
    else:
        # 创建传统的指纹模型实例
        N = params['N']  # fingerprint_dict的大小
        dim = params['dim']
        layer_hidden = params['layer_hidden']
        layer_output = params['layer_output']
        
        model = MolecularGraphNeuralNetwork(N, dim, layer_hidden, layer_output).to(device)
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, params

def standardize_property_values(properties, mean, std):
    """使用给定的均值和标准差对属性值进行标准化"""
    return [(p - mean) / std for p in properties]

def parse_smiles(smiles, property_value):
    """将SMILES字符串解析为模型输入格式"""
    global skipped_molecules_count, skipped_molecules_details
    try:
        # 复用preprocess.py中的函数
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            skipped_molecules_count += 1
            skipped_molecules_details.append({
                'smiles': smiles,
                'reason': 'Invalid SMILES - could not be parsed by RDKit',
                'property_value': property_value
            })
            return None
            
        mol = Chem.AddHs(mol)
        atoms = pp.create_atoms(mol, atom_dict)
        molecular_size = len(atoms)
        i_jbond_dict = pp.create_ijbonddict(mol, bond_dict)
        fingerprints = pp.extract_fingerprints(radius, atoms, i_jbond_dict,
                                              fingerprint_dict, edge_dict)
        adjacency = np.float32(Chem.GetAdjacencyMatrix(mol))
        
        # 检查维度一致性
        if len(fingerprints) != molecular_size or adjacency.shape[0] != molecular_size or adjacency.shape[1] != molecular_size:
            # 增加跳过的分子计数
            skipped_molecules_count += 1
            skipped_molecules_details.append({
                'smiles': smiles,
                'reason': f'Dimension mismatch - fingerprints length: {len(fingerprints)}, adjacency matrix shape: {adjacency.shape}, molecular size: {molecular_size}',
                'property_value': property_value
            })
            return None
            
        return (smiles, fingerprints, adjacency, molecular_size, float(property_value))
    except Exception as e:
        skipped_molecules_count += 1
        skipped_molecules_details.append({
            'smiles': smiles,
            'reason': f'Exception during processing: {str(e)}',
            'property_value': property_value
        })
        return None

def parse_smiles_for_visnet(smiles, property_value, filename):
    """将SMILES字符串解析为VisNet模型输入格式"""
    global skipped_molecules_count, mol_cache, skipped_molecules_details
    try:
        # 使用统一的原子类型映射
        types = ATOM_TYPES
        
        # 使用缓存处理 SMILES
        from utils.molecule import MoleculeCache
        if mol_cache is None:
            # 使用数据集名称作为tag，便于复用
            mol_cache = MoleculeCache(f"visnet_predict_{filename}")
            # 输出缓存统计信息
            print(f"初始化分子缓存: visnet_predict_{filename}")
        
        # 在处理前获取缓存统计信息
        initial_hits = mol_cache.cache_hits
        initial_misses = mol_cache.cache_misses
        
        result = mol_cache.process_smiles(smiles, types)
        
        # 处理后输出缓存命中情况
        if mol_cache.cache_hits > initial_hits:
            pass
        elif mol_cache.cache_misses > initial_misses:
            print(f"缓存未命中，新处理分子: {smiles}")
        
        # 检查处理结果
        if result is not None and len(result) == 5:
            x, z, pos, edge_index, edge_attr = result
            if x is None or z is None or pos is None or edge_index is None or edge_attr is None:
                skipped_molecules_count += 1
                skipped_molecules_details.append({
                    'smiles': smiles,
                    'reason': 'VisNet processing returned None values',
                    'property_value': property_value
                })
                return None
            
            # 确保z和pos是torch.Tensor类型
            if not isinstance(z, torch.Tensor):
                z = torch.tensor(z, dtype=torch.long)
            if not isinstance(pos, torch.Tensor):
                pos = torch.tensor(pos, dtype=torch.float)
            
            # 构造原子序数张量（VisNet需要的格式）
            atomic_numbers = z
            
            # pos已经是torch.Tensor格式，可以直接使用
            positions = pos
            
            return (smiles, atomic_numbers, positions, float(property_value))
        else:
            skipped_molecules_count += 1
            skipped_molecules_details.append({
                'smiles': smiles,
                'reason': f'VisNet processing failed - result format invalid, got {type(result)} with length {len(result) if result is not None else "None"}',
                'property_value': property_value
            })
            return None
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {str(e)}")
        skipped_molecules_count += 1
        skipped_molecules_details.append({
            'smiles': smiles,
            'reason': f'Exception during VisNet processing: {str(e)}',
            'property_value': property_value
        })
        return None

def predict_batch(model, dataset_batch, device, is_visnet=False):
    """对一批数据进行预测"""
    if len(dataset_batch) == 0:
        return [], [], []
    
    # 检查是否包含Norman_SusDat_ID (通过检查元组长度)
    has_norman_id = False
    sample_length = len(dataset_batch[0])
    # VisNet模型通常有4个基本元素，传统模型有5个基本元素
    # 如果长度比基本元素多1，则说明包含Norman_SusDat_ID
    if is_visnet and sample_length > 4:
        has_norman_id = True
    elif not is_visnet and sample_length > 5:
        has_norman_id = True
    
    # 检查模型类型
    from models.visnet import VisNetV1
    is_visnet_v1 = isinstance(model, VisNetV1)
    
    if is_visnet and not is_visnet_v1:
        # VisNet模型的处理逻辑
        if has_norman_id:
            smiles_list, atomic_numbers_list, positions_list, properties_list, norman_ids = zip(*dataset_batch)
        else:
            smiles_list, atomic_numbers_list, positions_list, properties_list = zip(*dataset_batch)
        
        # 合并批次数据
        all_atomic_numbers = []
        all_positions = []
        batch_indices = []
        
        atom_count = 0
        for i in range(len(atomic_numbers_list)):
            num_atoms = atomic_numbers_list[i].shape[0]
            all_atomic_numbers.append(atomic_numbers_list[i])
            all_positions.append(positions_list[i])
            batch_indices.extend([i] * num_atoms)
            atom_count += num_atoms
        
        # 转换为张量
        atomic_numbers = torch.cat(all_atomic_numbers, dim=0).to(device)
        positions = torch.cat([p.unsqueeze(0) if p.dim() == 1 else p for p in all_positions], dim=0).to(device)
        batch_tensor = torch.tensor(batch_indices, dtype=torch.long).to(device)
        correct_values = torch.tensor(properties_list, dtype=torch.float).to(device)
        
        # 进行预测
        with torch.no_grad():
            predicted_values, _ = model(atomic_numbers, positions, batch_tensor)
        
        # 确保返回的是CPU上的numpy数组或普通数值，与base_model保持一致
        if isinstance(predicted_values, torch.Tensor):
            predicted_values = predicted_values.detach().cpu()
        if isinstance(correct_values, torch.Tensor):
            correct_values = correct_values.detach().cpu()
        
        return smiles_list, predicted_values.flatten(), correct_values
    elif is_visnet_v1:
        # VisNetV1模型的处理逻辑
        from torch_geometric.data import Data
        # 根据实际数据格式解包
        if has_norman_id:
            if len(dataset_batch[0]) == 5:  # smiles, atomic_numbers, positions, property, norman_id
                smiles_list, atomic_numbers_list, positions_list, properties_list, norman_ids = zip(*dataset_batch)
            else:  # smiles, atomic_numbers, positions, molecular_size, property, norman_id
                smiles_list, atomic_numbers_list, positions_list, molecular_sizes_list, properties_list, norman_ids = zip(*dataset_batch)
        else:
            if len(dataset_batch[0]) == 4:  # smiles, atomic_numbers, positions, property
                smiles_list, atomic_numbers_list, positions_list, properties_list = zip(*dataset_batch)
            else:  # smiles, atomic_numbers, positions, molecular_size, property
                smiles_list, atomic_numbers_list, positions_list, molecular_sizes_list, properties_list = zip(*dataset_batch)
        
        predicted_values = []
        correct_values = []
        
        # VisNetV1需要逐个处理样本
        for i in range(len(atomic_numbers_list)):
            try:
                # 确保atomic_numbers和positions是PyTorch张量
                if not isinstance(atomic_numbers_list[i], torch.Tensor):
                    atomic_numbers_tensor = torch.tensor(atomic_numbers_list[i], dtype=torch.long)
                else:
                    atomic_numbers_tensor = atomic_numbers_list[i]
                    
                if not isinstance(positions_list[i], torch.Tensor):
                    positions_tensor = torch.tensor(positions_list[i], dtype=torch.float)
                else:
                    positions_tensor = positions_list[i]
                
                # 检查张量是否为空
                if atomic_numbers_tensor.numel() == 0 or positions_tensor.numel() == 0:
                    print(f"Warning: Empty tensor for SMILES {smiles_list[i]}")
                    continue
                    
                # 创建Data对象
                data = Data(
                    z=atomic_numbers_tensor.to(device),
                    pos=positions_tensor.to(device),
                    batch=torch.zeros(positions_tensor.size(0), dtype=torch.long).to(device)
                )
                
                # 进行预测
                with torch.no_grad():
                    pred = model(data)
                    predicted_values.append(pred.cpu())
                    correct_values.append(torch.tensor(properties_list[i]))
            except Exception as e:
                print(f"Error processing SMILES {smiles_list[i]}: {str(e)}")
                continue
        
        if len(predicted_values) == 0:
            return [], [], []
            
        predicted_values = torch.cat(predicted_values, dim=0)
        correct_values = torch.stack(correct_values)
        
        # 确保返回的是CPU上的numpy数组或普通数值，与base_model保持一致
        if isinstance(predicted_values, torch.Tensor):
            predicted_values = predicted_values.detach().cpu()
        if isinstance(correct_values, torch.Tensor):
            correct_values = correct_values.detach().cpu()
        
        return smiles_list, predicted_values.flatten(), correct_values.flatten()
    else:
        # 传统指纹模型的处理逻辑
        # 准备批次数据
        if has_norman_id:
            smiles_list, fingerprints_list, adjacencies_list, molecular_sizes_list, properties_list, norman_ids = zip(*dataset_batch)
        else:
            smiles_list, fingerprints_list, adjacencies_list, molecular_sizes_list, properties_list = zip(*dataset_batch)
        
        # 转换为张量并移动到设备
        fingerprints_list = [torch.LongTensor(f).to(device) for f in fingerprints_list]
        adjacencies_list = [torch.FloatTensor(a).to(device) for a in adjacencies_list]
        properties_list = [torch.FloatTensor([p]).to(device) for p in properties_list]  # 修改为一维张量
        
        # 构造输入数据
        data_batch = [smiles_list, fingerprints_list, adjacencies_list, molecular_sizes_list, properties_list]
        
        # 进行预测
        with torch.no_grad():
            Smiles, predicted_values, correct_values = model.forward_regressor(data_batch, train=False)
        
        # 确保返回的是CPU上的numpy数组或普通数值，与base_model保持一致
        if isinstance(predicted_values, torch.Tensor):
            predicted_values = predicted_values.detach().cpu()
        if isinstance(correct_values, torch.Tensor):
            correct_values = correct_values.detach().cpu()
        
        return Smiles, predicted_values.flatten(), correct_values.flatten()

def predict_dataset(model, dataset, device, batch_size=256, is_visnet=False):
    """对整个数据集进行预测"""
    predictions = []
    N = len(dataset)
    
    # 判断是否包含Norman_SusDat_ID
    has_norman_id = False
    if N > 0:
        # 检查第一个样本的长度来判断是否包含Norman_SusDat_ID
        sample_length = len(dataset[0])
        # VisNet模型通常有4个基本元素，传统模型有5个基本元素
        # 如果长度比基本元素多1，则说明包含Norman_SusDat_ID
        if is_visnet and sample_length > 4:
            has_norman_id = True
        elif not is_visnet and sample_length > 5:
            has_norman_id = True
    
    for i in range(0, N, batch_size):
        batch = dataset[i:i+batch_size]
        Smiles, predicted_values, correct_values = predict_batch(model, batch, device, is_visnet)
        
        for smile, pred, true in zip(Smiles, predicted_values, correct_values):
            predictions.append((smile, pred, true))
    
    return predictions

def load_dictionaries(dict_dir, dataset_name):
    """加载训练时使用的字典"""
    # 首先尝试加载不带前缀的字典文件（这是训练时保存的格式）
    base_dict_files = ['atom_dict.pickle', 'bond_dict.pickle', 'edge_dict.pickle', 'fingerprint_dict.pickle']
    dicts = {}
    loaded_count = 0
    
    # 尝试直接在dict_dir中查找字典文件
    for dict_file in base_dict_files:
        dict_path = os.path.join(dict_dir, dict_file)
        if os.path.exists(dict_path):
            with open(dict_path, 'rb') as f:
                loaded_dict = pickle.load(f)
                dicts[dict_file.replace('.pickle', '')] = loaded_dict
                # 更新全局字典
                if 'atom_dict' in dict_file:
                    atom_dict.update(loaded_dict)
                elif 'bond_dict' in dict_file:
                    bond_dict.update(loaded_dict)
                elif 'edge_dict' in dict_file:
                    edge_dict.update(loaded_dict)
                elif 'fingerprint_dict' in dict_file:
                    fingerprint_dict.update(loaded_dict)
            print(f"Loaded dictionary: {dict_path}")
            loaded_count += 1
        else:
            print(f"Warning: Dictionary file {dict_path} not found.")
    
    # 如果基本字典文件未找到，尝试带数据集名称前缀的文件
    if loaded_count < len(base_dict_files):
        for dict_file in base_dict_files:
            dict_path = os.path.join(dict_dir, f"{dataset_name}-{dict_file}")
            if os.path.exists(dict_path):
                with open(dict_path, 'rb') as f:
                    loaded_dict = pickle.load(f)
                    dicts[dict_file.replace('.pickle', '')] = loaded_dict
                    # 更新全局字典
                    if 'atom_dict' in dict_file:
                        atom_dict.update(loaded_dict)
                    elif 'bond_dict' in dict_file:
                        bond_dict.update(loaded_dict)
                    elif 'edge_dict' in dict_file:
                        edge_dict.update(loaded_dict)
                    elif 'fingerprint_dict' in dict_file:
                        fingerprint_dict.update(loaded_dict)
                print(f"Loaded dictionary: {dict_path}")
                loaded_count += 1
            else:
                # 尝试在子目录中查找
                sub_dir_path = os.path.join(dict_dir, dataset_name)
                if os.path.exists(sub_dir_path):
                    dict_path = os.path.join(sub_dir_path, f"{dataset_name}-{dict_file}")
                    if os.path.exists(dict_path):
                        with open(dict_path, 'rb') as f:
                            loaded_dict = pickle.load(f)
                            dicts[dict_file.replace('.pickle', '')] = loaded_dict
                            # 更新全局字典
                            if 'atom_dict' in dict_file:
                                atom_dict.update(loaded_dict)
                            elif 'bond_dict' in dict_file:
                                bond_dict.update(loaded_dict)
                            elif 'edge_dict' in dict_file:
                                edge_dict.update(loaded_dict)
                            elif 'fingerprint_dict' in dict_file:
                                fingerprint_dict.update(loaded_dict)
                        print(f"Loaded dictionary: {dict_path}")
                        loaded_count += 1

    if loaded_count == len(base_dict_files):
        print(f"Successfully loaded all {loaded_count} dictionaries")
    else:
        print(f"Warning: Only loaded {loaded_count} out of {len(base_dict_files)} dictionaries")
    
    return dicts

def save_simple_predictions(predictions, output_path, filename):
    """保存简单的预测结果到CSV文件，不进行任何评估"""
    # 创建一个简单的数据框，只包含SMILES和预测值
    simple_predictions = [(smile, pred) for smile, pred, _ in predictions]
    pred_df = pd.DataFrame(simple_predictions, columns=['SMILES', 'Predicted'])
    pred_df.to_csv(output_path, index=False)
    print(f"Saved simple predictions to {output_path}")

def save_skipped_molecules_report(skipped_details, output_dir, filename):
    """保存跳过的分子报告到CSV文件"""
    if not skipped_details:
        return
        
    # 创建DataFrame并保存到CSV
    skipped_df = pd.DataFrame(skipped_details)
    skipped_filename = filename.replace('.csv', '_skipped_molecules.csv')
    skipped_filepath = os.path.join(output_dir, skipped_filename)
    skipped_df.to_csv(skipped_filepath, index=False)
    print(f"Saved skipped molecules report to {skipped_filepath}")


def main():
    global skipped_molecules_count, skipped_molecules_details
    parser = argparse.ArgumentParser(description='Predict using trained GNN model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--params_path', type=str, required=True, help='Path to the training parameters JSON file')
    parser.add_argument('--dict_dir', type=str, default='/home/data2/rhj/project/gnn/gnn-1/data/MMF-3', 
                        help='Directory containing dictionaries')
    
    parser.add_argument('--dataset_name', type=str, default='MMF-GNN', 
                        help='Dataset name prefix for dictionary files')
    
    parser.add_argument('--input_dir', type=str, default='/home/data2/rhj/project/gnn/gnn-1/data/MMF-3',
                        help='Directory containing input CSV files')
    parser.add_argument('--output_dir', type=str, default='./predictions',
                        help='Directory to save prediction results')
    
    # 
    parser.add_argument('--target_column', type=str, help='Target property column name in CSV')
    parser.add_argument('--smiles_column', type=str, default='SMILES', 
                        help='SMILES column name in CSV')
    parser.add_argument('--filter_column', type=str, default="auto",  # put others to close filter
                        help='Column name to filter data (e.g., Uncertainty_RTI_pos)')
    parser.add_argument('--filter_value', type=str, default='Covered by chemical space of the model',
                        help='Value to filter data (default: "Covered by chemical space of the model")')
    
    #     
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for prediction')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to predict (default: None, which means all samples)')
    parser.add_argument('--files_to_process', type=str, default=None,
                        help='Comma-separated list of files to process (default: auto-determined based on model)')
    parser.add_argument('--dataset_type', type=str, default=None,
                        help='Dataset type to filter SMILES (e.g., "test" to use test set file). '
                             'If specified, only molecules in the corresponding test set file will be predicted.')
    parser.add_argument('--simple_prediction', action='store_true',
                        help='Enable simple prediction mode that only outputs predictions to CSV without evaluation')
    parser.add_argument('--use_cpu', action='store_true',
                        help='Force using CPU instead of GPU for prediction')

    args = parser.parse_args()
    
    # 创建带时间戳的输出目录
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    output_dir_with_timestamp = os.path.join(args.output_dir, f"prediction_{timestamp}")
    os.makedirs(output_dir_with_timestamp, exist_ok=True)
    
    # 检查CUDA可用性
    if args.use_cpu:
        device = torch.device('cpu')
        print('Using device: CPU (forced)')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')
    
    # 加载字典
    print("Loading dictionaries...")
    # 从参数文件中获取训练时使用的数据集名称
    with open(args.params_path, 'r') as f:
        params = json.load(f)
        trained_dataname = params.get('dataname', args.dataset_name)
    
    # 检查是否是VisNet模型
    is_visnet = (params.get('model_type') in ['visnet', 'visnet_v1', 'visnet_v2'] or 
                params.get('visnet', False) or 
                params.get('visnet_v1', False) or 
                params.get('visnet_v2', False) or 
                params.get('visnet_mfe', False))
    
    # 只有非VisNet模型需要加载字典
    if not is_visnet:
        # 使用训练时的数据集名称加载字典
        loaded_dicts = load_dictionaries(args.dict_dir, trained_dataname)
        if not loaded_dicts:
            print("Error: No dictionaries loaded. Cannot proceed with prediction.")
            return
    
    # 加载模型
    print("Loading model...")
    model, params = load_model(args.model_path, args.params_path, device)
    print(f"Model loaded with parameters: {params}")
    
    # 根据训练参数中的dataname确定要处理的文件和目标列
    trained_dataname = params.get('dataname', '')
    print(f"Model was trained on: {trained_dataname}")
    
    # 如果没有指定目标列，则根据训练数据自动确定
    if args.target_column is None:
        if 'pos' in trained_dataname:
            args.target_column = 'Pred_RTI_Positive_ESI'
        elif 'neg' in trained_dataname:
            args.target_column = 'Pred_RTI_Negative_ESI'
        else:
            args.target_column = 'RT'  # 默认值
    
    print(f"Using target column: {args.target_column}")
    
    # 确定相应的测试文件
    files_to_process = []
    if args.files_to_process:
        # 如果通过参数指定了目标文件，则使用指定的文件
        files_to_process = [f.strip() for f in args.files_to_process.split(',')]
    elif 'pos' in trained_dataname:
        # 如果模型是在pos数据上训练的，使用对应的not文件进行测试
        files_to_process = ['MMF_GNN_pos.csv']
    elif 'neg' in trained_dataname:
        # 如果模型是在neg数据上训练的，使用对应的not文件进行测试
        files_to_process = ['MMF_GNN_neg.csv']
    else:
        files_to_process = ['MMF_GNN_pos.csv']  # 默认处理训练集

    # 自动设置过滤列和值
    if args.filter_column == "auto":
        if 'RTI_pos' in trained_dataname:
            args.filter_column = 'Uncertainty_RTI_pos'
        elif 'RTI_neg' in trained_dataname:
            args.filter_column = 'Uncertainty_RTI_neg'
    else:
        args.filter_column = None  # 不进行过滤
    
    print(f"Files to process: {files_to_process}")
    if args.filter_column:
        print(f"Filtering by {args.filter_column} = {args.filter_value}")
    
    # 处理每个文件
    for filename in files_to_process:
        # 重置跳过的分子计数器和详情列表
        skipped_molecules_count = 0
        skipped_molecules_details = []
        filepath = os.path.join(args.input_dir, filename)
        if not os.path.exists(filepath):
            print(f"Warning: File {filepath} not found, skipping...")
            continue
        
        print(f"Processing {filename}...")
        
        # 读取数据
        df = pd.read_csv(filepath, low_memory=False)
        print(f"Loaded {len(df)} entries from {filename}")
        
        # 如果指定了dataset_type，则根据对应的测试集文件过滤数据
        if args.dataset_type:
            test_set_file = None
            if args.dataset_type == 'test':
                # 根据训练数据名称确定测试集文件路径
                if 'pos' in trained_dataname:
                    test_set_file = os.path.join(args.dict_dir, 'MMF_GNN_pos', 'MMF_GNN_pos_test_set.txt')
                elif 'neg' in trained_dataname:
                    test_set_file = os.path.join(args.dict_dir, 'MMF_GNN_neg', 'MMF_GNN_neg_test_set.txt')
            
            print('😀', test_set_file, trained_dataname)
            
            if test_set_file and os.path.exists(test_set_file):
                print(f"Filtering data based on test set file: {test_set_file}")
                # 读取测试集中的SMILES
                test_smiles = set()
                with open(test_set_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            parts = line.strip().split('\t')
                            if len(parts) >= 1:
                                test_smiles.add(parts[0])
                
                # 过滤数据框，只保留测试集中的SMILES
                original_count = len(df)
                df = df[df[args.smiles_column].isin(test_smiles)]
                filtered_count = len(df)
                print(f"Filtered data from {original_count} to {filtered_count} entries based on test set")
            else:
                print(f"Warning: Test set file not found for dataset_type '{args.dataset_type}'")
        
        # 根据filter_column和filter_value过滤数据
        if args.filter_column and args.filter_column in df.columns:
            original_count = len(df)
            df = df[df[args.filter_column] == args.filter_value]
            filtered_count = len(df)
            print(f"Filtered data from {original_count} to {filtered_count} entries "
                  f"based on {args.filter_column} = '{args.filter_value}'")
        
        # 检查是否存在Norman_SusDat_ID列
        has_norman_id = 'Norman_SusDat_ID' in df.columns
        
        # 解析SMILES
        dataset = []
        # 使用tqdm显示处理进度
        rows = list(df.iterrows())
        if args.max_samples is not None:
            rows = rows[:args.max_samples]
            
        if is_visnet:
            iter_rows = tqdm(rows, total=len(rows), desc=f"Processing {filename}")
        else:
            iter_rows = rows
            
        for idx, row in iter_rows:
            smiles = row[args.smiles_column]
            property_value = row[args.target_column]
            if is_visnet:
                parsed_data = parse_smiles_for_visnet(smiles, property_value, filename)
            else:
                parsed_data = parse_smiles(smiles, property_value)
            if parsed_data is not None:
                # 如果存在Norman_SusDat_ID，则将其添加到数据中
                if has_norman_id:
                    norman_id = row['Norman_SusDat_ID']
                    if is_visnet:
                        # VisNet模型数据格式: (smiles, atomic_numbers, positions, property_value)
                        parsed_data = (parsed_data[0], parsed_data[1], parsed_data[2], parsed_data[3], norman_id)
                    else:
                        # 传统模型数据格式: (smiles, fingerprints, adjacency, molecular_size, property_value)
                        parsed_data = (parsed_data[0], parsed_data[1], parsed_data[2], parsed_data[3], parsed_data[4], norman_id)
                dataset.append(parsed_data)
            
            # 如果设置了最大样本数并且已达到限制，则停止处理更多样本
            if args.max_samples is not None and len(dataset) >= args.max_samples:
                print(f"Reached maximum number of samples ({args.max_samples}), stopping processing")
                break
        
        print(f"Parsed {len(dataset)} valid molecules from {filename} (skipped {skipped_molecules_count} molecules)")
        
        # 保存跳过的分子报告
        save_skipped_molecules_report(skipped_molecules_details, output_dir_with_timestamp, filename)
        
        if len(dataset) == 0:
            print(f"No valid molecules in {filename}, skipping prediction...")
            continue
        
        # 如果训练参数中包含标准化信息，则对目标值进行标准化
        if 'property_mean' in params and 'property_std' in params:
            print(f"Standardizing property values using mean={params['property_mean']:.4f}, std={params['property_std']:.4f}")
            # 保存原始数据用于后续计算标准化前后的MAE
            if is_visnet:
                if has_norman_id:
                    original_predictions_data = [(data[0], data[3]) for data in dataset]  # smiles, property_value
                else:
                    original_predictions_data = [(data[0], data[3]) for data in dataset]  # smiles, property_value
            else:
                if has_norman_id:
                    original_predictions_data = [(data[0], data[4]) for data in dataset]  # smiles, property_value
                else:
                    original_predictions_data = [(data[0], data[4]) for data in dataset]  # smiles, property_value
            
            # 更新数据集中的属性值
            if is_visnet:
                # VisNet模型的数据格式处理
                standardized_dataset = []
                for data in dataset:
                    if has_norman_id:
                        smiles, atomic_numbers, positions, property_value, norman_id = data
                        standardized_property = (property_value - params['property_mean']) / params['property_std']
                        standardized_dataset.append((smiles, atomic_numbers, positions, standardized_property, norman_id))
                    else:
                        smiles, atomic_numbers, positions, property_value = data
                        standardized_property = (property_value - params['property_mean']) / params['property_std']
                        standardized_dataset.append((smiles, atomic_numbers, positions, standardized_property))
                dataset = standardized_dataset
            else:
                # 传统模型的数据格式处理
                standardized_dataset = []
                for data in dataset:
                    if has_norman_id:
                        smiles, fingerprints, adjacency, molecular_size, property_value, norman_id = data
                        standardized_property = (property_value - params['property_mean']) / params['property_std']
                        standardized_dataset.append((smiles, fingerprints, adjacency, molecular_size, standardized_property, norman_id))
                    else:
                        smiles, fingerprints, adjacency, molecular_size, property_value = data
                        standardized_property = (property_value - params['property_mean']) / params['property_std']
                        standardized_dataset.append((smiles, fingerprints, adjacency, molecular_size, standardized_property))
                dataset = standardized_dataset
        else:
            original_predictions_data = None
        
        # 进行预测
        print(f"Predicting {len(dataset)} molecules...")
        predictions = predict_dataset(model, dataset, device, args.batch_size, is_visnet)
        
        # 计算标准化MAE（如果适用）
        standardized_mae = None
        if 'property_mean' in params and 'property_std' in params:
            # 在反标准化之前计算标准化的MAE
            if len(predictions) > 0:
                std_preds = np.array([float(p[1]) for p in predictions])
                std_actuals = np.array([float(p[2]) for p in predictions])
                
                # 处理张量到numpy的转换
                if torch.is_tensor(std_preds):
                    std_preds = std_preds.detach().cpu().numpy()
                if torch.is_tensor(std_actuals):
                    std_actuals = std_actuals.detach().cpu().numpy()
                
                # 检查NaN或无穷大值
                valid_mask = np.isfinite(std_preds) & np.isfinite(std_actuals)
                if np.all(valid_mask) and len(std_preds) > 0:
                    standardized_mae = np.mean(np.abs(std_preds - std_actuals))

        # 如果进行了标准化，则将预测结果反标准化并计算原始MAE
        original_mae = None
        if 'property_mean' in params and 'property_std' in params and original_predictions_data is not None:
            print("De-standardizing predictions...")
            # 创建原始真实值的映射
            original_true_dict = dict(original_predictions_data)
            
            # 反标准化预测值和实际值，并计算原始MAE
            destandardized_predictions = []
            original_pred_values = []
            original_true_values = []
            
            for smile, pred, true in predictions:
                # 确保pred和true是标量值
                if isinstance(pred, torch.Tensor):
                    pred = pred.item() if pred.numel() == 1 else pred.detach().cpu().numpy()
                if isinstance(true, torch.Tensor):
                    true = true.item() if true.numel() == 1 else true.detach().cpu().numpy()
                
                # 反标准化预测值
                destandardized_pred = float(pred) * params['property_std'] + params['property_mean']
                # 使用原始数据中的真实值，而不是反标准化后的预测值
                if smile in original_true_dict:
                    destandardized_true = float(original_true_dict[smile])
                else:
                    continue  # 跳过无法找到原始真实值的样本
                
                destandardized_predictions.append((smile, destandardized_pred, destandardized_true))
                original_pred_values.append(destandardized_pred)
                original_true_values.append(destandardized_true)
            
            # 计算原始尺度MAE
            if len(original_pred_values) > 0 and len(original_true_values) > 0:
                # 检查NaN或无穷大值
                valid_mask = np.isfinite(original_pred_values) & np.isfinite(original_true_values)
                if np.all(valid_mask) and len(original_pred_values) > 0:
                    original_mae = np.mean(np.abs(np.array(original_pred_values) - np.array(original_true_values)))
            
            predictions = destandardized_predictions
        else:
            original_mae = None

        # 保存预测结果
        output_filename = filename.replace('.csv', '_predictions.csv')
        output_filepath = os.path.join(output_dir_with_timestamp, output_filename)
        
        if args.simple_prediction:
            # 简单预测模式：只保存预测值，不进行评估
            simple_output_filename = filename.replace('.csv', '_simple_predictions.csv')
            simple_output_filepath = os.path.join(output_dir_with_timestamp, simple_output_filename)
            save_simple_predictions(predictions, simple_output_filepath, filename)
        else:
            # 标准模式：保存预测值并进行评估
            # 创建包含Norman_SusDat_ID的DataFrame（如果存在）
            if has_norman_id:
                # 创建SMILES到Norman_SusDat_ID的映射
                smile_to_norman_id = {}
                if is_visnet:
                    for data in dataset:
                        if len(data) > 4:  # 包含Norman_SusDat_ID
                            smile_to_norman_id[data[0]] = data[4]
                else:
                    for data in dataset:
                        if len(data) > 5:  # 包含Norman_SusDat_ID
                            smile_to_norman_id[data[0]] = data[5]
                
                # 构建预测数据，包含Norman_SusDat_ID
                pred_data = []
                for smile, pred, true in predictions:
                    norman_id = smile_to_norman_id.get(smile, '')
                    pred_data.append((norman_id, smile, pred, true))
                
                pred_df = pd.DataFrame(pred_data, columns=['Norman_SusDat_ID', 'SMILES', 'Predicted', 'Actual'])
            else:
                pred_df = pd.DataFrame(predictions, columns=['SMILES', 'Predicted', 'Actual'])
            
            pred_df.to_csv(output_filepath, index=False)
            print(f"Saved predictions to {output_filepath}")
            
            # 筛选并保存误差较大的预测结果
            if len(predictions) > 0:
                preds = np.array([float(p[1]) for p in predictions])
                actuals = np.array([float(p[2]) for p in predictions])
                # 计算相对误差百分比
                epsilon = 1e-8
                relative_errors = np.abs((preds - actuals) / np.maximum(np.abs(actuals), epsilon)) * 100
                # 筛选误差大于40%的项目
                high_error_mask = relative_errors > 40
                high_error_predictions = [p for i, p in enumerate(predictions) if high_error_mask[i]]
                
                if len(high_error_predictions) > 0:
                    print(f"\n发现 {len(high_error_predictions)} 个误差大于40%的项目")
                    # 保存高误差预测到单独文件
                    high_error_filename = filename.replace('.csv', '_high_error_predictions.csv')
                    high_error_filepath = os.path.join(output_dir_with_timestamp, high_error_filename)
                    
                    # 添加误差列
                    high_error_data = []
                    for pred in high_error_predictions:
                        smile, predicted, actual = pred
                        error = abs(predicted - actual)
                        relative_error_pct = (error / max(abs(actual), epsilon)) * 100
                        high_error_data.append((smile, predicted, actual, error, relative_error_pct))
                    
                    high_error_df = pd.DataFrame(
                        high_error_data, 
                        columns=['SMILES', 'Predicted', 'Actual', 'Absolute_Error', 'Relative_Error_Percent']
                    )
                    high_error_df.to_csv(high_error_filepath, index=False)
                    print(f"高误差预测已保存到 {high_error_filepath}")
                
                # 计算评估指标
                if len(predictions) > 0:
                    evaluation_metrics = calculate_evaluation_metrics(predictions, filename)
                    mae = evaluation_metrics["mae"]
                    rmse = evaluation_metrics["rmse"]
                    r2 = evaluation_metrics["r2"]
                    pcc = evaluation_metrics["pcc"]
                    
                    if original_mae is not None:
                        print(f"  Original Scale MAE: {original_mae:.4f}")
                    if standardized_mae is not None:
                        print(f"  Standardized MAE: {standardized_mae:.4f}")
                    
                    # 计算基于不同误差阈值的准确率
                    accuracy_metrics = calculate_accuracy_metrics(predictions)
                    
                    # 生成误差分布饼图
                    plot_error_distribution(predictions, filename, output_dir_with_timestamp)
                    
                    # 生成并保存评估指标报告
                    metrics_data = generate_metrics_report(
                        evaluation_metrics=evaluation_metrics,
                        accuracy_metrics=accuracy_metrics,
                        original_mae=original_mae,
                        standardized_mae=standardized_mae,
                        df=df,
                        dataset=dataset,
                        skipped_molecules_count=skipped_molecules_count,
                        args=args,
                        trained_dataname=trained_dataname,
                        filename=filename
                    )
                        
                    metrics_filename = filename.replace('.csv', '_metrics.json')
                    metrics_filepath = os.path.join(output_dir_with_timestamp, metrics_filename)
                    save_metrics_report(metrics_data, metrics_filepath)
                    
                    # 如果是VisNet模型且使用了缓存，输出缓存统计信息
                    if is_visnet and mol_cache is not None:
                        cache_stats = mol_cache.get_stats()
                        print(f"Molecule cache statistics:")
                        print(f"  Hits: {cache_stats['hits']}")
                        print(f"  Misses: {cache_stats['misses']}")
                        print(f"  Hit rate: {cache_stats['hit_rate']:.2%}")
                        print(f"  Cache size: {cache_stats['cache_size']}")
                        print(f"  Failed molecules: {cache_stats['failed_count']}")

if __name__ == '__main__':
    main()