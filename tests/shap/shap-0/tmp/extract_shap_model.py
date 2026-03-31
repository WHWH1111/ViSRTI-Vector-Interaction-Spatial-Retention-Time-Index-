import sys
import os
import json
import torch
import torch.nn as nn

# 获取当前文件的目录并将其添加到sys.path中，以确保可以进行绝对导入
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from models.visnet_v2_shap import VisNetV2SHAP


def extract_shap_model(model_path, training_params_path, output_path, physchem_dim=4):
    """
    从训练好的VisNetV2模型中提取权重并创建SHAP模型
    
    Args:
        model_path: 原始模型文件路径
        training_params_path: 训练参数文件路径
        output_path: 输出SHAP模型路径
        physchem_dim: 物化特征维度（默认为4）
    """
    # 读取训练参数
    with open(training_params_path, 'r') as f:
        training_params = json.load(f)
    
    # 检查模型类型
    if training_params['model_type'] != 'visnet_v2':
        raise ValueError("该脚本仅支持VisNetV2模型")
    
    # 创建SHAP模型，使用指定的物化特征维度
    shap_model = VisNetV2SHAP(
        node_feature_dim=training_params['visnet_v2_node_feature_dim'],
        physchem_feature_dim=physchem_dim,  # 使用指定维度而不是训练参数中的维度
        graph_hidden_dim=training_params['visnet_v2_graph_hidden_dim'],
        physchem_hidden_dim=training_params['visnet_v2_physchem_hidden_dim'],  # 保持与原始模型一致
        fusion_hidden_dims=training_params['visnet_v2_fusion_hidden_dims'],
        dropout_rate=0.0  # SHAP模型不需要dropout
    )
    
    # 加载原始模型权重
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    original_model_state = torch.load(model_path, map_location=device)
    
    # 获取SHAP模型状态字典
    shap_model_state = shap_model.state_dict()
    
    # 更新SHAP模型状态字典
    updated_state = {}
    
    # 复制VisNet权重（直接匹配）
    for key in original_model_state.keys():
        if key.startswith('visnet.'):
            updated_state[key] = original_model_state[key]
    
    # 映射其他权重（名称完全匹配的直接复制）
    direct_mapping = [
        # 图特征处理器
        'graph_processor.0.weight',
        'graph_processor.0.bias',
        'graph_processor.1.weight',
        'graph_processor.1.bias',
        'graph_processor.4.weight',
        'graph_processor.4.bias',
        'graph_processor.5.weight',
        'graph_processor.5.bias',
        
        # 融合网络
        'fusion_net.0.weight',
        'fusion_net.0.bias',
        'fusion_net.1.weight',
        'fusion_net.1.bias',
        'fusion_net.4.weight',
        'fusion_net.4.bias',
        'fusion_net.5.weight',
        'fusion_net.5.bias',
        'fusion_net.8.weight',
        'fusion_net.8.bias',
        'fusion_net.10.weight',
        'fusion_net.10.bias',
    ]
    
    # 直接复制匹配的权重
    for key in direct_mapping:
        if key in original_model_state:
            updated_state[key] = original_model_state[key]
        else:
            print(f"警告: 权重 {key} 在原始模型中未找到")

    # 物化特征处理器权重需要特殊处理
    # 原始模型: Linear(4->64) -> BatchNorm1d(64) -> ReLU -> Dropout -> Linear(64->128) -> ReLU
    # SHAP模型: Linear(4->64) -> BatchNorm1d(64) -> ReLU -> Dropout -> Linear(64->128) -> ReLU
    physchem_mapping = [
        # 第一层: Linear(4->64) 和 BatchNorm1d(64)
        ('physchem_processor.0.weight', 'physchem_processor.0.weight'),
        ('physchem_processor.0.bias', 'physchem_processor.0.bias'),
        ('physchem_processor.1.weight', 'physchem_processor.1.weight'),
        ('physchem_processor.1.bias', 'physchem_processor.1.bias'),
        # 第二层: Linear(64->128)
        ('physchem_processor.4.weight', 'physchem_processor.4.weight'),
        ('physchem_processor.4.bias', 'physchem_processor.4.bias'),
    ]
    
    # 映射物化特征处理器权重
    for shap_key, original_key in physchem_mapping:
        if original_key in original_model_state:
            original_weight = original_model_state[original_key]
            if shap_key in shap_model_state:
                shap_weight = shap_model_state[shap_key]
                
                # 检查维度是否匹配
                if original_weight.shape == shap_weight.shape:
                    updated_state[shap_key] = original_weight
                else:
                    print(f"维度不匹配 {original_key} ({original_weight.shape}) -> {shap_key} ({shap_weight.shape})")
                    # 尝试适配维度（截取或填充）
                    if len(original_weight.shape) == 2:  # 权重矩阵
                        min_out_features = min(original_weight.shape[0], shap_weight.shape[0])
                        min_in_features = min(original_weight.shape[1], shap_weight.shape[1])
                        adapted_weight = torch.zeros_like(shap_weight)
                        adapted_weight[:min_out_features, :min_in_features] = original_weight[:min_out_features, :min_in_features]
                        updated_state[shap_key] = adapted_weight
                    elif len(original_weight.shape) == 1:  # 偏置向量
                        min_features = min(original_weight.shape[0], shap_weight.shape[0])
                        adapted_weight = torch.zeros_like(shap_weight)
                        adapted_weight[:min_features] = original_weight[:min_features]
                        updated_state[shap_key] = adapted_weight
            else:
                print(f"SHAP模型中未找到键: {shap_key}")
        else:
            print(f"原始模型中未找到键: {original_key}")
    
    # 加载权重到SHAP模型
    shap_model.load_state_dict(updated_state, strict=False)
    
    # 保存SHAP模型
    torch.save(shap_model.state_dict(), output_path)
    print(f"SHAP模型已保存到: {output_path}")
    
    return shap_model


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='从VisNetV2模型提取SHAP模型')
    parser.add_argument('--model-path', type=str, required=True,
                        help='原始模型文件路径')
    parser.add_argument('--training-params-path', type=str, required=True,
                        help='训练参数文件路径')
    parser.add_argument('--output-path', type=str, required=True,
                        help='输出SHAP模型路径')
    parser.add_argument('--physchem-dim', type=int, default=4,
                        help='物化特征维度（默认为4）')
    
    args = parser.parse_args()
    
    extract_shap_model(args.model_path, args.training_params_path, args.output_path, args.physchem_dim)


if __name__ == "__main__":
    main()