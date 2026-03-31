import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
import os
import argparse
from collections import Counter
import datetime
import json

def calculate_metrics(y_true, y_pred, property_mean=None, property_std=None):
    """
    计算评估指标
    
    Args:
        y_true (array): 真实值
        y_pred (array): 预测值
        property_mean (float, optional): 属性均值，用于标准化
        property_std (float, optional): 属性标准差，用于标准化
        
    Returns:
        tuple: (r2, mae, normalized_mae) R2、MAE和标准化MAE指标
    """
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    # 如果提供了均值和标准差，则计算标准化MAE
    normalized_mae = None
    if property_mean is not None and property_std is not None:
        # 标准化真实值和预测值
        y_true_normalized = (y_true - property_mean) / property_std
        y_pred_normalized = (y_pred - property_mean) / property_std
        normalized_mae = mean_absolute_error(y_true_normalized, y_pred_normalized)
    
    return r2, mae, normalized_mae

def load_data(filepath):
    """
    加载预测数据
    
    Args:
        filepath (str): 预测结果文件路径
        
    Returns:
        DataFrame: 包含预测数据的DataFrame
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"预测结果文件不存在: {filepath}")
    
    try:
        data = pd.read_csv(filepath)
        return data
    except Exception as e:
        print(f"读取文件时出错: {e}")
        raise

def plot_all_samples(y_true, y_pred, output_file='all_samples_residuals.png', params_file=None):
    """
    绘制所有样本的残差图
    
    Args:
        y_true (array): 真实值
        y_pred (array): 预测值
        output_file (str): 输出文件名
        params_file (str, optional): 训练参数文件路径
    """
    residuals = y_true - y_pred
    
    # 初始化标准化参数
    property_mean = None
    property_std = None
    
    # 如果提供了参数文件，则从中读取均值和标准差
    if params_file and os.path.exists(params_file):
        try:
            with open(params_file, 'r') as f:
                params = json.load(f)
                property_mean = params.get('property_mean')
                property_std = params.get('property_std')
                print(f"Loaded normalization parameters from {params_file}: mean={property_mean}, std={property_std}")
        except Exception as e:
            print(f"Warning: Failed to load parameters from {params_file}: {e}")
    
    # 计算指标
    r2, mae, normalized_mae = calculate_metrics(y_true, y_pred, property_mean, property_std)
    n_samples = len(y_true)
    
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # 绘制散点图
    ax.scatter(y_pred, residuals, alpha=0.6, color='gray', edgecolors='white', s=50)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Residual=0')
    ax.set_xlabel('Predicted Values', fontsize=12)
    ax.set_ylabel('Residuals', fontsize=12)
    ax.set_title('Prediction Residual Scatter Plot (All Samples)', fontsize=14, fontweight='bold', pad=20)
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_facecolor('#f8f9fa')
    
    # 添加指标文本
    if normalized_mae is not None:
        metrics_text = f'Items = {n_samples}\nR2 = {r2:.3f}\nMAE = {mae:.3f}\nNormalized MAE = {normalized_mae:.3f}'
    else:
        metrics_text = f'Items = {n_samples}\nR2 = {r2:.3f}\nMAE = {mae:.3f}'
        
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 添加整体标题
    fig.suptitle(f'Residual Analysis (R2={r2:.3f})', fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"所有样本残差图已保存至: {output_file}")
    plt.close()

def plot_emphasized_category(y_true, y_pred, tags, target_tag, output_file, params_file=None):
    """
    绘制强调某个分类的残差图
    
    Args:
        y_true (array): 真实值
        y_pred (array): 预测值
        tags (list): 每个样本的标签列表
        target_tag (str): 需要强调的目标标签
        output_file (str): 输出文件名
        params_file (str, optional): 训练参数文件路径
    """
    residuals = y_true - y_pred
    
    # 默认所有点为灰色
    colors = ['gray' for _ in range(len(y_pred))]
    
    # 找出目标标签的索引
    target_indices = []
    for i, tag_str in enumerate(tags):
        if target_tag in tag_str:
            colors[i] = 'red'
            target_indices.append(i)
    
    # 初始化标准化参数
    property_mean = None
    property_std = None
    
    # 如果提供了参数文件，则从中读取均值和标准差
    if params_file and os.path.exists(params_file):
        try:
            with open(params_file, 'r') as f:
                params = json.load(f)
                property_mean = params.get('property_mean')
                property_std = params.get('property_std')
        except Exception as e:
            print(f"Warning: Failed to load parameters from {params_file}: {e}")
    
    # 计算该类别的指标
    if target_indices:
        target_y_true = y_true[target_indices]
        target_y_pred = y_pred[target_indices]
        r2, mae, normalized_mae = calculate_metrics(target_y_true, target_y_pred, property_mean, property_std)
        n_samples = len(target_indices)
    else:
        r2, mae, normalized_mae = 0, 0, None
        n_samples = 0
    
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # 绘制散点图
    ax.scatter(y_pred, residuals, alpha=0.6, color=colors, edgecolors='white', s=50)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Residual=0')
    ax.set_xlabel('Predicted Values', fontsize=12)
    ax.set_ylabel('Residuals', fontsize=12)
    ax.set_title(f'Prediction Residual Scatter Plot (Emphasizing: {target_tag})', fontsize=14, fontweight='bold', pad=20)
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_facecolor('#f8f9fa')
    
    # 添加指标文本
    if normalized_mae is not None:
        metrics_text = f'Items = {n_samples}\nR2 = {r2:.3f}\nMAE = {mae:.3f}\nNormalized MAE = {normalized_mae:.3f}'
    else:
        metrics_text = f'Items = {n_samples}\nR2 = {r2:.3f}\nMAE = {mae:.3f}'
        
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 添加整体标题
    overall_r2 = r2_score(y_true, y_pred)
    fig.suptitle(f'Residual Analysis (R2={overall_r2:.3f})', fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"{target_tag} 类别强调图已保存至: {output_file}")
    plt.close()

def get_top_tags(compound_tags, top_n=10):
    """
    获取出现频率最高的前N个标签
    
    Args:
        compound_tags (list): 化合物标签列表
        top_n (int): 返回前N个标签
        
    Returns:
        list: 出现频率最高的前N个标签
    """
    # 统计所有标签的出现次数
    all_tags = []
    for tag_str in compound_tags:
        tags = tag_str.split(',')
        all_tags.extend(tags)
    
    tag_counter = Counter(all_tags)
    top_tags = [tag for tag, _ in tag_counter.most_common(top_n)]
    return top_tags

"""
# pre

# run

1. pos
python scripts\plt\plot_prediction_residuals.py -i predictions\visnet-v2\pos-3-best-prediction_20251126-191743\test_MMF_GNN_pos_predictions_with_tags.csv -o pos-test -p "log\visnet-v2\pos-3-mask(koc)-train_20251126-172959_dim48_layerH6_layerO6_batch64_lr0.0001_iter150\training_params.json"

2. neg
python scripts\plt\plot_prediction_residuals.py -i predictions\visnet-v2\neg-3-best-prediction_20251128-210839\test_MMF_GNN_neg_predictions_with_tags.csv -o neg-test -p "log\visnet-v2\neg-3-mask(koc)-train_20251128-162233_dim48_layerH6_layerO6_batch64_lr0.0001_iter150\training_params.json"
"""


def main():
    parser = argparse.ArgumentParser(description='绘制预测残差散点图')
    parser.add_argument('--input', '-i', type=str, required=True, help='预测结果文件路径')
    parser.add_argument('--base-output-dir', '-b', type=str, default='results/residual_plots', help='基础输出图像文件目录')
    parser.add_argument('--output-dir', '-o', type=str, help='输出图像文件目录')
    parser.add_argument('--params-file', '-p', type=str, help='训练参数文件路径，用于标准化MAE计算')
    
    args = parser.parse_args()
    
    try:
        # 创建输出目录
        if args.output_dir is None:
            args.output_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        output_dir_with_timestamp = os.path.join(args.base_output_dir, args.output_dir)
        os.makedirs(output_dir_with_timestamp, exist_ok=True)
        
        # 加载数据
        data = load_data(args.input)
        
        # 提取需要的列
        y_true = data['Actual'].values
        y_pred = data['Predicted'].values
        compound_tags = data['Compound_Tags'].values
        
        # 检查是否有足够的数据
        if len(y_true) == 0:
            print("错误: 没有有效的预测数据")
            return
        
        print(f"加载了 {len(y_true)} 个有效预测样本")
        
        # 绘制所有样本的残差图
        all_samples_file = os.path.join(output_dir_with_timestamp, 'all_samples_residuals.png')
        plot_all_samples(y_true, y_pred, all_samples_file, args.params_file)
        
        # 获取最常见的10个标签
        top_tags = get_top_tags(compound_tags, 10)
        print(f"最常见的10个标签: {top_tags}")
        
        # 为每个主要标签绘制强调图
        for i, tag in enumerate(top_tags):
            output_file = os.path.join(output_dir_with_timestamp, f'{i+1:02d}_{tag}_emphasized_residuals.png')
            plot_emphasized_category(y_true, y_pred, compound_tags, tag, output_file, args.params_file)
            
        print(f"\n所有残差图已保存至目录: {output_dir_with_timestamp}")
        
    except Exception as e:
        print(f"处理过程中出错: {e}")
        raise

if __name__ == '__main__':
    main()