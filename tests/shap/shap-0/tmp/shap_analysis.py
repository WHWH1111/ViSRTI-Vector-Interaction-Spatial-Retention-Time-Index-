import sys
import os
import json
import torch
import numpy as np
import argparse

# 获取当前文件的目录并将其添加到sys.path中，以确保可以进行绝对导入
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 导入模型
from models.visnet_v2_shap import VisNetV2SHAP

# 导入工具函数
from core import load_datasets, preprocess_visnet_data


def load_shap_model(model_path, training_params_path, device):
    """
    加载训练好的SHAP模型
    """
    # 读取训练参数
    with open(training_params_path, 'r') as f:
        training_params = json.load(f)
    
    model = VisNetV2SHAP(
        node_feature_dim=training_params['visnet_v2_node_feature_dim'],
        physchem_feature_dim=4,  # 使用4维物化特征
        graph_hidden_dim=training_params['visnet_v2_graph_hidden_dim'],
        physchem_hidden_dim=training_params['visnet_v2_physchem_hidden_dim'],
        fusion_hidden_dims=training_params['visnet_v2_fusion_hidden_dims'],
        dropout_rate=0.0
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def prepare_physchem_data(dataset, preprocessed_data, sample_size=10):
    """
    准备物化特征数据用于SHAP分析
    """
    # 选择样本子集进行分析
    indices = np.random.choice(len(dataset), min(sample_size, len(dataset)), replace=False)
    
    # 提取物化特征
    physchem_features = []
    valid_indices = []
    smiles_list = []
    
    for i in indices:
        smiles = dataset[i][0]  # 假设SMILES在第一个位置
        if smiles in preprocessed_data and preprocessed_data[smiles]['physchem_features'] is not None:
            # 确保是torch.Tensor类型
            feat = preprocessed_data[smiles]['physchem_features']
            if isinstance(feat, np.ndarray):
                feat = torch.from_numpy(feat)
            physchem_features.append(feat)
            valid_indices.append(i)
            smiles_list.append(smiles)
    
    if len(physchem_features) == 0:
        raise ValueError("没有找到有效的物化特征数据")
    
    physchem_features = torch.stack(physchem_features).float()
    print(f"使用 {len(physchem_features)} 个样本进行SHAP分析")
    
    return physchem_features, smiles_list


class SimpleSHAPWrapper(torch.nn.Module):
    """
    简化版包装模型用于SHAP分析
    """
    def __init__(self, model, fixed_graph_inputs):
        super(SimpleSHAPWrapper, self).__init__()
        self.model = model
        self.fixed_z = fixed_graph_inputs['z']
        self.fixed_pos = fixed_graph_inputs['pos']
        self.fixed_batch = fixed_graph_inputs['batch']
    
    def forward(self, physchem_features):
        """
        前向传播，变化物化特征
        """
        # 确保所有张量在同一设备上
        device = physchem_features.device
        z = self.fixed_z.to(device)
        pos = self.fixed_pos.to(device)
        batch = self.fixed_batch.to(device)
        
        # 确保输入是float32类型
        physchem_features = physchem_features.float()
        
        # 获取批量大小
        batch_size = physchem_features.shape[0]
        
        # 如果批量大小大于1，需要扩展图数据
        if batch_size > 1:
            # 重复图数据以匹配批量大小
            num_atoms = z.shape[0]
            z = z.unsqueeze(0).repeat(batch_size, 1).view(-1)
            pos = pos.unsqueeze(0).repeat(batch_size, 1, 1).view(-1, 3)
            batch = torch.arange(batch_size, device=device).repeat_interleave(num_atoms)
        
        pred, _ = self.model(z, pos, batch, physchem_features)
        # 确保输出是二维的
        if pred.dim() == 1:
            pred = pred.unsqueeze(1)
        return pred


def compute_simple_shap_values(model, physchem_features, graph_data, device):
    """
    计算简化版SHAP值
    """
    # 将物化特征移到设备上
    physchem_features = physchem_features.to(device)
    
    # 创建固定图输入
    fixed_inputs = {
        'z': graph_data['z'],
        'pos': graph_data['pos'],
        'batch': torch.zeros(graph_data['z'].shape[0], dtype=torch.long)
    }
    
    # 创建包装模型
    wrapped_model = SimpleSHAPWrapper(model, fixed_inputs)
    wrapped_model = wrapped_model.to(device)
    
    # 使用背景数据（取前5个样本）
    background = physchem_features[:5].to(device)
    
    # 创建SHAP解释器
    print("创建SHAP解释器...")
    import shap
    explainer = shap.DeepExplainer(wrapped_model, background)
    
    # 计算SHAP值
    print("计算SHAP值...")
    shap_values = explainer.shap_values(physchem_features.to(device))
    
    return shap_values


def analyze_shap_results(shap_values, physchem_features, output_dir):
    """
    分析并保存SHAP结果为文本格式
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 转换为numpy数组
    if isinstance(shap_values, torch.Tensor):
        shap_values = shap_values.cpu().detach().numpy()
    if isinstance(physchem_features, torch.Tensor):
        physchem_features = physchem_features.cpu().detach().numpy()
    
    # 物化特征名称
    feature_names = ['LogP', 'MW', 'TPSA', 'nRotB']
    
    # 计算特征重要性
    shap_abs_mean = np.mean(np.abs(shap_values), axis=0)
    
    # 打印特征重要性
    print("\n特征重要性排序 (基于平均|SHAP值|):")
    feature_importance = [(name, float(importance)) for name, importance in zip(feature_names, shap_abs_mean)]
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    for i, (name, importance) in enumerate(feature_importance):
        print(f"{i+1}. {name}: {importance:.6f}")
    
    # 保存特征重要性到JSON文件
    importance_dict = {name: importance for name, importance in feature_importance}
    with open(os.path.join(output_dir, 'feature_importance.json'), 'w') as f:
        json.dump(importance_dict, f, indent=2)
    
    # 保存详细的SHAP值到文件
    shap_details = {
        'feature_names': feature_names,
        'shap_values': shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values,
        'physchem_features': physchem_features.tolist(),
        'mean_abs_shap': shap_abs_mean.tolist()
    }
    
    with open(os.path.join(output_dir, 'shap_details.json'), 'w') as f:
        json.dump(shap_details, f, indent=2)
    
    return feature_importance


def main():
    parser = argparse.ArgumentParser(description='Fixed VisNetV2 SHAP Analysis for PhysChem Features')
    
    # 数据和模型参数
    parser.add_argument('--model-path', type=str, required=True,
                        help='训练好的SHAP模型路径')
    parser.add_argument('--training-params-path', type=str, required=True,
                        help='训练参数文件路径')
    parser.add_argument('--data-path', type=str, default='./data/MMF-3/',
                        help='数据路径')
    parser.add_argument('--dataset-name', type=str, default='MMF_GNN_neg',
                        help='数据集名称')
    parser.add_argument('--output-dir', type=str, default='./shap_results',
                        help='SHAP结果输出目录')
    parser.add_argument('--sample-size', type=int, default=10,
                        help='用于SHAP分析的样本数量')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    print("加载模型...")
    model = load_shap_model(args.model_path, args.training_params_path, device)
    
    # 准备数据（简化处理）
    print("准备数据...")
    
    # 模拟训练参数
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
    physchem_features, smiles_list = prepare_physchem_data(
        dataset_test_filtered, test_preprocessed_data, args.sample_size)
    
    # 获取一个样本的图数据作为固定输入
    sample_smiles = smiles_list[0]
    graph_data = test_preprocessed_data[sample_smiles]
    
    # 计算SHAP值
    print("开始SHAP分析...")
    shap_values = compute_simple_shap_values(model, physchem_features, graph_data, device)
    
    # 分析结果
    print("分析结果...")
    feature_importance = analyze_shap_results(shap_values, physchem_features, args.output_dir)
    
    print(f"\nSHAP分析完成，结果保存在 {args.output_dir}")
    print("\n特征重要性排序:")
    for i, (name, importance) in enumerate(feature_importance):
        print(f"{i+1}. {name}: {importance:.6f}")


if __name__ == "__main__":
    main()