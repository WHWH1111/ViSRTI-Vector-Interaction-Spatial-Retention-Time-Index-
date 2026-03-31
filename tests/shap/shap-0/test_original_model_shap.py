import sys
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import shap
import pandas as pd
import argparse

# 获取当前文件的目录并将其添加到sys.path中，以确保可以进行绝对导入
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    # 导入模型
    from models.visnet_v2 import VisNetV2
    # 导入工具函数
    from core import load_datasets, preprocess_visnet_data
except ImportError:
    print("无法导入模型")


def load_original_model(model_path, training_params_path, device):
    """
    加载训练好的原始VisNetV2模型
    """
    # 读取训练参数
    with open(training_params_path, 'r') as f:
        training_params = json.load(f)
        
    print(training_params)
    
    model = VisNetV2(
        node_feature_dim=training_params['visnet_v2_node_feature_dim'],
        physchem_feature_dim=training_params['visnet_v2_physchem_feature_dim'],
        toxicity_feature_dim=training_params['visnet_v2_toxicity_feature_dim'],
        chromato_feature_dim=training_params['visnet_v2_chromato_feature_dim'],
        graph_hidden_dim=training_params['visnet_v2_graph_hidden_dim'],
        physchem_hidden_dim=training_params['visnet_v2_physchem_hidden_dim'],
        toxicity_hidden_dim=training_params['visnet_v2_toxicity_hidden_dim'],
        chromato_hidden_dim=training_params['visnet_v2_chromato_hidden_dim'],
        fusion_hidden_dims=training_params['visnet_v2_fusion_hidden_dims'],
        dropout_rate=training_params.get('dropout', 0.0),
        use_attention=training_params['visnet_v2_use_attention'],
        use_gating=training_params['visnet_v2_use_gating']
    ).to(device)
    
    # 加载模型权重，处理键不匹配问题
    state_dict = torch.load(model_path, map_location=device)
    
    # 过滤掉fusion_net相关的键（这些是动态构建的）
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fusion_net.')}
    
    # 加载过滤后的权重
    model.load_state_dict(filtered_state_dict, strict=False)
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


def prepare_graph_data(preprocessed_data, smiles_list, sample_size=5):
    """
    准备多个图数据用于SHAP分析
    """
    graph_data_list = []
    valid_smiles = []
    
    for smiles in smiles_list[:sample_size]:  # 限制图样本数量以控制计算复杂度
        graph_data = preprocessed_data[smiles]
        # 确保图数据是正确的格式
        if 'z' in graph_data and 'pos' in graph_data:
            graph_data_list.append({
                'z': graph_data['z'],
                'pos': graph_data['pos']
            })
            valid_smiles.append(smiles)
        elif 'atom_types' in graph_data and 'coordinates' in graph_data:
            graph_data_list.append({
                'z': graph_data['atom_types'],
                'pos': graph_data['coordinates']
            })
            valid_smiles.append(smiles)
            
    print(f"准备了 {len(graph_data_list)} 个图结构样本用于SHAP分析")
    return graph_data_list, valid_smiles


class OriginalModelSHAPWrapper(torch.nn.Module):
    """
    包装原始VisNetV2模型用于SHAP分析
    """
    def __init__(self, model, fixed_graph_inputs):
        super(OriginalModelSHAPWrapper, self).__init__()
        self.model = model
        self.fixed_z = fixed_graph_inputs['z']
        self.fixed_pos = fixed_graph_inputs['pos']
        self.fixed_batch = fixed_graph_inputs['batch']
    
    def forward(self, physchem_features):
        """
        前向传播，只变化物化特征
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
        
        # 调用原始模型的forward方法，只传入图特征和物化特征
        pred, _ = self.model(z, pos, batch, physchem_features=physchem_features)
        
        # 确保输出是二维的，形状为[batch_size, 1]，这对于SHAP分析很重要
        if pred.dim() == 0:
            pred = pred.unsqueeze(0).unsqueeze(1)
        elif pred.dim() == 1:
            pred = pred.unsqueeze(1)
        elif pred.dim() > 2:
            pred = pred.squeeze()
            if pred.dim() == 0:
                pred = pred.unsqueeze(0).unsqueeze(1)
            elif pred.dim() == 1:
                pred = pred.unsqueeze(1)
                
        return pred


class MultiGraphSHAPWrapper(torch.nn.Module):
    """
    包装原始VisNetV2模型用于多图结构SHAP分析
    """
    def __init__(self, model, graph_data_list):
        super(MultiGraphSHAPWrapper, self).__init__()
        self.model = model
        self.graph_data_list = graph_data_list
        
    def forward(self, physchem_features):
        """
        前向传播，变化物化特征并在多个图结构上运行
        """
        device = physchem_features.device
        batch_size = physchem_features.shape[0]
        num_graphs = len(self.graph_data_list)
        
        # 为每个图结构重复物化特征
        # 将[batch_size, feature_dim]扩展为[num_graphs * batch_size, feature_dim]
        expanded_physchem_features = physchem_features.repeat(num_graphs, 1)
        
        # 构建所有图的组合批次
        all_z = []
        all_pos = []
        all_batch = []
        
        node_offset = 0
        batch_offset = 0
        for i, graph_data in enumerate(self.graph_data_list):
            z = graph_data['z'].to(device)
            pos = graph_data['pos'].to(device)
            
            # 对于每个图，重复处理batch_size次
            num_atoms = z.shape[0]
            
            # 重复图数据以匹配物化特征批次
            z_repeated = z.unsqueeze(0).repeat(batch_size, 1).view(-1)
            pos_repeated = pos.unsqueeze(0).repeat(batch_size, 1, 1).view(-1, 3)
            batch_repeated = torch.arange(batch_size, device=device).repeat_interleave(num_atoms)
            batch_repeated += batch_offset  # 调整batch索引
            
            all_z.append(z_repeated)
            all_pos.append(pos_repeated)
            all_batch.append(batch_repeated)
            
            # 更新偏移量
            node_offset += batch_size * num_atoms
            batch_offset += batch_size
            
        # 合并所有图数据
        combined_z = torch.cat(all_z, dim=0)
        combined_pos = torch.cat(all_pos, dim=0)
        combined_batch = torch.cat(all_batch, dim=0)
        
        # 调用原始模型
        pred, _ = self.model(combined_z, combined_pos, combined_batch, 
                           physchem_features=expanded_physchem_features)
        
        # 重塑输出，将结果按图结构分组并求平均
        # pred形状为[num_graphs * batch_size, 1]
        pred = pred.view(num_graphs, batch_size)
        pred = pred.mean(dim=0)  # 在图结构维度上求平均
        
        # 确保输出是正确的形状[batch_size, 1]
        return pred.unsqueeze(1)


def compute_shap_values(model, physchem_features, graph_data, device):
    """
    计算SHAP值
    """
    # 将物化特征移到设备上
    physchem_features = physchem_features.to(device)
    
    # 创建固定图输入
    fixed_inputs = {
        'z': graph_data['z'],
        'pos': graph_data['pos'],
        'batch': torch.zeros(graph_data['z'].shape[0], dtype=torch.long)
    }
    
    # 检查数据是否正确
    print(f"Fixed z shape: {fixed_inputs['z'].shape}")
    print(f"Fixed pos shape: {fixed_inputs['pos'].shape}")
    print(f"Physchem features shape: {physchem_features.shape}")
    
    # 创建包装模型
    wrapped_model = OriginalModelSHAPWrapper(model, fixed_inputs)
    wrapped_model = wrapped_model.to(device)
    
    # 测试模型是否能正常运行
    with torch.no_grad():
        test_output = wrapped_model(physchem_features[:2])
        print(f"Model output shape: {test_output.shape}")
        print(f"Model output: {test_output}")
    
    # 使用背景数据（取前5个样本）
    background = physchem_features[:5].to(device)
    
    # 创建SHAP解释器
    print("创建SHAP解释器...")
    explainer = shap.DeepExplainer(wrapped_model, background)
    
    # 计算SHAP值，禁用additivity检查
    print("计算SHAP值...")
    shap_values = explainer.shap_values(physchem_features.to(device), check_additivity=False)
    
    # 确保shap_values是正确的形状
    print(f"SHAP values shape: {shap_values.shape}")
    
    return shap_values


def compute_shap_values_multi_graph(model, physchem_features, graph_data_list, device):
    """
    使用多个图结构计算SHAP值
    """
    # 将物化特征移到设备上
    physchem_features = physchem_features.to(device)
    
    # 创建包装模型
    wrapped_model = MultiGraphSHAPWrapper(model, graph_data_list)
    wrapped_model = wrapped_model.to(device)
    
    # 测试模型是否能正常运行
    with torch.no_grad():
        test_output = wrapped_model(physchem_features[:2])
        print(f"Model output shape: {test_output.shape}")
        print(f"Model output: {test_output}")
    
    # 使用背景数据（取前5个样本）
    background = physchem_features[:5].to(device)
    
    # 创建SHAP解释器
    print("创建SHAP解释器...")
    explainer = shap.DeepExplainer(wrapped_model, background)
    
    # 计算SHAP值，禁用additivity检查
    print("计算SHAP值...")
    shap_values = explainer.shap_values(physchem_features.to(device), check_additivity=False)
    
    # 确保shap_values是正确的形状
    print(f"SHAP values shape: {shap_values.shape}")
    
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
    feature_importance = [(name, float(importance.item())) for name, importance in zip(feature_names, shap_abs_mean)]
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


def plot_shap_results(shap_values, physchem_features, output_dir):
    """
    绘制SHAP分析图，参考draw-demo-1.py的绘图方式
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 转换为numpy数组
    if isinstance(shap_values, torch.Tensor):
        shap_values = shap_values.cpu().detach().numpy()
    if isinstance(physchem_features, torch.Tensor):
        physchem_features = physchem_features.cpu().detach().numpy()
    
    # 物化特征名称
    feature_names = ['LogP', 'MW', 'TPSA', 'nRotB']
    
    # 如果shap_values是3D数组，我们需要将其展平为2D
    if len(shap_values.shape) == 3 and shap_values.shape[2] == 1:
        shap_values = shap_values.squeeze(-1)
    
    # 创建DataFrame用于蜂群图（顶部x轴 - 特征值）
    bee_data = []
    for i, feature_name in enumerate(feature_names):
        for value in physchem_features[:, i]:
            bee_data.append({'Category': feature_name, 'Value': float(value)})
    bee_df = pd.DataFrame(bee_data)
    
    # 创建DataFrame用于SHAP值散点图（底部x轴 - SHAP值）
    shap_data = []
    for i, feature_name in enumerate(feature_names):
        for value in shap_values[:, i]:
            shap_data.append({'Category': feature_name, 'SHAP': float(value)})
    shap_df = pd.DataFrame(shap_data)
    
    # 创建颜色映射对象
    from matplotlib.colors import LinearSegmentedColormap
    colors = [(0, 'blue'), (1, 'red')]
    cmap_name = 'my_colormap'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors)
    
    # 创建图形和主坐标轴
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # 绘制SHAP值散点图（底部x轴 - SHAP值）
    sns.stripplot(data=shap_df, x='SHAP', y='Category', hue='Category', ax=ax1,
                  palette=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
                  size=6, alpha=0.7, jitter=0.3, edgecolor='black', linewidth=0.5,
                  marker='s', legend=False)  # 使用方形标记区分SHAP值
    
    # 设置底部x轴的标签和样式（SHAP值）
    ax1.set_xlabel('SHAP Values (Feature Importance)', fontsize=14, fontweight='bold', color='#A23B72')
    ax1.tick_params(axis='x', labelcolor='#A23B72')
    ax1.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    # 在SHAP图中添加垂直线标记SHAP=0
    ax1.axvline(x=0, color='red', linestyle='-', alpha=0.5, linewidth=1)
    
    # 创建顶部x轴的坐标轴（共享y轴）
    ax2 = ax1.twiny()
    
    # 绘制蜂群图（顶部x轴 - 特征值）
    sns.stripplot(data=bee_df, x='Value', y='Category', hue='Category', ax=ax2, 
                  palette=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
                  size=6, alpha=0.7, jitter=0.3, edgecolor='black', linewidth=0.5,
                  legend=False)
    
    # 设置顶部x轴的标签和样式（特征值分布）
    ax2.set_xlabel('Feature Value Distribution (Bee Swarm)', fontsize=14, fontweight='bold', color='#2E86AB')
    ax2.tick_params(axis='x', labelcolor='#2E86AB')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 设置y轴标签
    ax1.set_ylabel('Feature Categories', fontsize=14, fontweight='bold')
    
    # 调整两个x轴的位置
    ax1.xaxis.set_ticks_position('bottom')
    ax1.xaxis.set_label_position('bottom')
    ax2.xaxis.set_ticks_position('top')
    ax2.xaxis.set_label_position('top')
    
    # 设置坐标轴范围
    shap_min, shap_max = shap_df['SHAP'].min(), shap_df['SHAP'].max()
    # 对于较小的SHAP值，使用更紧凑的范围
    shap_range = max(abs(shap_min), abs(shap_max))
    ax1.set_xlim(-shap_range * 1.2, shap_range * 1.2)
    
    feature_min, feature_max = bee_df['Value'].min(), bee_df['Value'].max()
    feature_range = feature_max - feature_min
    ax2.set_xlim(feature_min - feature_range * 0.1, feature_max + feature_range * 0.1)
    
    # 增加y轴标签间距，解决过于紧凑的问题
    ax1.set_ylim(-0.5, len(feature_names) - 0.5)
    ax2.set_ylim(-0.5, len(feature_names) - 0.5)
    
    # 添加标题
    plt.title('Feature Analysis: Feature Value Distribution vs SHAP Importance', 
              fontsize=16, fontweight='bold', pad=40)
    
    # 创建自定义图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=8, label='Feature Values', alpha=0.7),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
               markersize=8, label='SHAP Values', alpha=0.7),
        Line2D([0], [0], color='red', linestyle='-', alpha=0.5, 
               label='SHAP=0 Baseline', linewidth=2)
    ]
    ax1.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
    
    # 添加右侧颜色条
    norm = plt.Normalize(bee_df['Value'].min(), bee_df['Value'].max())
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    sm.set_array([])  # only needed for matplotlib < 3.1
    
    # 添加颜色条到右侧，使用更小的pad值使其更靠近主图
    cbar = fig.colorbar(sm, ax=ax1, orientation='vertical', aspect=40, pad=0.08)
    cbar.ax.set_ylabel('Feature value', rotation=270, labelpad=15, fontsize=14, fontweight='bold')
    cbar.ax.text(0.5, 1.02, 'High', transform=cbar.ax.transAxes, ha='center', va='bottom', fontsize=12, fontweight='bold')
    cbar.ax.text(0.5, -0.02, 'Low', transform=cbar.ax.transAxes, ha='center', va='top', fontsize=12, fontweight='bold')
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_dual_axis_plot.pdf"))
    plt.savefig(os.path.join(output_dir, "shap_dual_axis_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"SHAP图表已保存到 {output_dir}")


def save_shap_data(shap_values, physchem_features, smiles_list, output_dir):
    """
    保存SHAP分析数据到JSON文件，以便后续绘图使用
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
    
    # 保存数据到JSON文件
    shap_data = {
        'feature_names': feature_names,
        'shap_values': shap_values.tolist(),
        'physchem_features': physchem_features.tolist(),
        'smiles_list': smiles_list
    }
    
    with open(os.path.join(output_dir, 'shap_data.json'), 'w') as f:
        json.dump(shap_data, f, indent=2)
    
    print(f"SHAP数据已保存到 {os.path.join(output_dir, 'shap_data.json')}")


def load_shap_data(output_dir):
    """
    从JSON文件加载SHAP分析数据
    """
    with open(os.path.join(output_dir, 'shap_data.json'), 'r') as f:
        shap_data = json.load(f)
    
    feature_names = shap_data['feature_names']
    shap_values = np.array(shap_data['shap_values'])
    physchem_features = np.array(shap_data['physchem_features'])
    smiles_list = shap_data['smiles_list']
    
    return shap_values, physchem_features, feature_names, smiles_list


def run_shap_analysis(args):
    """
    运行SHAP分析并将结果保存到文件
    """
    # 设置参数
    model_path = args.model_path
    training_params_path = args.training_params_path
    data_path = args.data_path
    dataset_name = args.dataset_name
    output_dir = args.output_dir
    sample_size = args.sample_size
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    print("加载模型...")
    model = load_original_model(model_path, training_params_path, device)
    
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
        args_obj, data_path, dataset_name, max_data=None)
    
    # VisNet系列模型需要特殊的预处理
    train_preprocessed_data, dev_preprocessed_data, test_preprocessed_data, \
    dataset_train_filtered, dataset_dev_filtered, dataset_test_filtered = preprocess_visnet_data(
        args_obj, dataset_train, dataset_dev, dataset_test, dataset_name, "visnet_train_", "visnet_test_")
    
    # 从训练数据集中采样用于SHAP分析
    physchem_features, smiles_list = prepare_physchem_data(
        dataset_train_filtered, train_preprocessed_data, sample_size)
    
    # 获取多个样本的图数据作为输入，使用训练集中的样本
    graph_data_list, graph_smiles_list = prepare_graph_data(
        train_preprocessed_data, smiles_list, sample_size=min(5, len(smiles_list)))
    
    # 计算SHAP值（使用多个图结构）
    print("开始SHAP分析...")
    shap_values = compute_shap_values_multi_graph(model, physchem_features, graph_data_list, device)
    
    # 保存SHAP数据
    print("保存SHAP数据...")
    save_shap_data(shap_values, physchem_features, smiles_list, output_dir)
    
    # 分析结果
    print("分析结果...")
    feature_importance = analyze_shap_results(shap_values, physchem_features, output_dir)
    
    print(f"\nSHAP分析完成，结果保存在 {output_dir}")
    print("\n特征重要性排序:")
    for i, (name, importance) in enumerate(feature_importance):
        print(f"{i+1}. {name}: {importance:.6f}")


def run_plot_shap(args):
    """
    从保存的数据中加载并绘制SHAP图
    """
    output_dir = args.output_dir
    
    # 加载SHAP数据
    print("加载SHAP数据...")
    shap_values, physchem_features, feature_names, smiles_list = load_shap_data(output_dir)
    
    # 绘制SHAP图
    print("绘制SHAP图...")
    plot_shap_results(shap_values, physchem_features, output_dir)
    
    print(f"\nSHAP图表已保存到 {output_dir}")


def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='SHAP Analysis for VisNetV2 Model')
    parser.add_argument('--mode', type=str, choices=['analyze', 'plot', 'both'], default='both',
                        help='运行模式: analyze (仅分析), plot (仅绘图), both (分析并绘图)')
    
    # 分析参数
    parser.add_argument('--model_path', type=str, 
                        default='./log/neg-train_20251028-110803_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/model.pt',
                        help='模型文件路径')
    parser.add_argument('--training_params_path', type=str,
                        default='./log/neg-train_20251028-110803_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/training_params.json',
                        help='训练参数文件路径')
    parser.add_argument('--data_path', type=str, default='./data/MMF-3/',
                        help='数据路径')
    parser.add_argument('--dataset_name', type=str, default='MMF_GNN_neg',
                        help='数据集名称')
    parser.add_argument('--sample_size', type=int, default=200,
                        help='用于SHAP分析的样本数量')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./test_shap_results',
                        help='输出目录')
    
    args = parser.parse_args()
    
    if args.mode == 'analyze':
        run_shap_analysis(args)
    elif args.mode == 'plot':
        run_plot_shap(args)
    elif args.mode == 'both':
        run_shap_analysis(args)
        run_plot_shap(args)


if __name__ == "__main__":
    main()
