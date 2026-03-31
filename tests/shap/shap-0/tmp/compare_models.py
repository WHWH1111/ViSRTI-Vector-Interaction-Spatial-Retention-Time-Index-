import sys
import os
import json
import torch
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 获取当前文件的目录并将其添加到sys.path中，以确保可以进行绝对导入
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 导入模型
from models.visnet_v2 import VisNetV2
from models.visnet_v2_shap import VisNetV2SHAP

# 导入工具函数
from core import load_datasets, preprocess_visnet_data


def load_original_model(model_path, training_params_path, device):
    """
    加载原始VisNetV2模型
    """
    # 读取训练参数
    with open(training_params_path, 'r') as f:
        training_params = json.load(f)
    
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
        use_attention=training_params.get('visnet_v2_use_attention', False),
        use_gating=training_params.get('visnet_v2_use_gating', False),
        dropout_rate=training_params.get('dropout', 0.0)
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()
    return model


def load_shap_model(model_path, training_params_path, device):
    """
    加载提取的SHAP模型
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


def prepare_test_data(dataset, preprocessed_data, sample_size=20):
    """
    准备测试数据
    """
    # 选择样本子集进行测试
    indices = np.random.choice(len(dataset), min(sample_size, len(dataset)), replace=False)
    
    # 提取数据
    test_data = []
    for i in indices:
        smiles = dataset[i][0]  # 假设SMILES在第一个位置
        if smiles in preprocessed_data:
            data = preprocessed_data[smiles]
            test_data.append((data, dataset[i][-1]))  # (预处理数据, 真实标签)
    
    return test_data


def compare_models(original_model, shap_model, test_data, device):
    """
    比较两个模型的预测效果
    """
    original_preds = []
    shap_preds = []
    true_labels = []
    
    with torch.no_grad():
        for data, label in test_data:
            # 获取数据
            z = data['z'].to(device)
            pos = data['pos'].to(device)
            batch = data.get('batch', torch.zeros(z.shape[0], dtype=torch.long)).to(device)
            physchem_features = data['physchem_features']
            
            # 确保physchem_features是tensor
            if isinstance(physchem_features, np.ndarray):
                physchem_features = torch.from_numpy(physchem_features)
            physchem_features = physchem_features.float().to(device)
            
            # 原始模型预测 (使用图和物化特征)
            original_pred, _ = original_model(z, pos, batch, physchem_features=physchem_features)
            original_pred = original_pred.cpu().numpy()
            
            # SHAP模型预测
            shap_pred, _ = shap_model(z, pos, batch, physchem_features)
            shap_pred = shap_pred.cpu().numpy()
            
            original_preds.append(original_pred)
            shap_preds.append(shap_pred)
            true_labels.append(label)
    
    # 转换为numpy数组
    original_preds = np.array(original_preds).flatten()
    shap_preds = np.array(shap_preds).flatten()
    true_labels = np.array(true_labels)
    
    # 计算指标
    # 1. 原始模型指标
    original_mae = mean_absolute_error(true_labels, original_preds)
    original_rmse = np.sqrt(mean_squared_error(true_labels, original_preds))
    original_r2 = pearsonr(true_labels, original_preds)[0] ** 2
    
    # 2. SHAP模型指标
    shap_mae = mean_absolute_error(true_labels, shap_preds)
    shap_rmse = np.sqrt(mean_squared_error(true_labels, shap_preds))
    shap_r2 = pearsonr(true_labels, shap_preds)[0] ** 2
    
    # 3. 模型间预测相关性
    pred_correlation = pearsonr(original_preds, shap_preds)[0]
    
    # 4. 预测差异
    pred_diff = np.abs(original_preds - shap_preds)
    mean_diff = np.mean(pred_diff)
    max_diff = np.max(pred_diff)
    
    return {
        'original_metrics': {
            'mae': original_mae,
            'rmse': original_rmse,
            'r2': original_r2
        },
        'shap_metrics': {
            'mae': shap_mae,
            'rmse': shap_rmse,
            'r2': shap_r2
        },
        'comparison': {
            'prediction_correlation': pred_correlation,
            'mean_prediction_difference': mean_diff,
            'max_prediction_difference': max_diff
        },
        'predictions': {
            'true_labels': true_labels,
            'original_preds': original_preds,
            'shap_preds': shap_preds
        }
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='比较原始模型和提取的SHAP模型')
    
    # 数据和模型参数
    parser.add_argument('--original-model-path', type=str, required=True,
                        help='原始模型文件路径')
    parser.add_argument('--shap-model-path', type=str, required=True,
                        help='SHAP模型文件路径')
    parser.add_argument('--training-params-path', type=str, required=True,
                        help='训练参数文件路径')
    parser.add_argument('--data-path', type=str, default='./data/MMF-3/',
                        help='数据路径')
    parser.add_argument('--dataset-name', type=str, default='MMF_GNN_neg',
                        help='数据集名称')
    parser.add_argument('--sample-size', type=int, default=20,
                        help='测试样本数量')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    print("加载原始模型...")
    original_model = load_original_model(args.original_model_path, args.training_params_path, device)
    
    print("加载SHAP模型...")
    shap_model = load_shap_model(args.shap_model_path, args.training_params_path, device)
    
    # 准备数据
    print("准备测试数据...")
    
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
    
    # 准备测试数据
    test_data = prepare_test_data(dataset_test_filtered, test_preprocessed_data, args.sample_size)
    print(f"准备了 {len(test_data)} 个测试样本")
    
    # 比较模型
    print("比较模型预测效果...")
    results = compare_models(original_model, shap_model, test_data, device)
    
    # 输出结果
    print("\n=== 模型效果比较结果 ===")
    print(f"\n原始模型指标:")
    print(f"  MAE: {results['original_metrics']['mae']:.4f}")
    print(f"  RMSE: {results['original_metrics']['rmse']:.4f}")
    print(f"  R²: {results['original_metrics']['r2']:.4f}")
    
    print(f"\nSHAP模型指标:")
    print(f"  MAE: {results['shap_metrics']['mae']:.4f}")
    print(f"  RMSE: {results['shap_metrics']['rmse']:.4f}")
    print(f"  R²: {results['shap_metrics']['r2']:.4f}")
    
    print(f"\n模型比较:")
    print(f"  预测相关性: {results['comparison']['prediction_correlation']:.4f}")
    print(f"  平均预测差异: {results['comparison']['mean_prediction_difference']:.6f}")
    print(f"  最大预测差异: {results['comparison']['max_prediction_difference']:.6f}")
    
    # 判断模型是否等效
    if results['comparison']['prediction_correlation'] > 0.99 and \
       results['comparison']['mean_prediction_difference'] < 0.01:
        print("\n✅ 模型效果基本一致，SHAP模型提取成功!")
    else:
        print("\n⚠️  模型效果存在差异，需要进一步检查!")
    
    # 保存详细结果
    output_dir = os.path.dirname(args.shap_model_path)
    results_file = os.path.join(output_dir, 'model_comparison_results.json')
    
    # 转换numpy数组为列表以便JSON序列化
    serializable_results = results.copy()
    serializable_results['predictions'] = {
        'true_labels': results['predictions']['true_labels'].tolist(),
        'original_preds': results['predictions']['original_preds'].tolist(),
        'shap_preds': results['predictions']['shap_preds'].tolist()
    }
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\n详细结果已保存到: {results_file}")


if __name__ == "__main__":
    main()