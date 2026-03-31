#!/usr/bin/env python3
"""
根据预测结果CSV文件生成R2和MAE指标的小提琴图

docs: docs\summary\violin_plot_generation.md

cli-example:
1. pos
cd d:\Projects\python\gnn-rt-1; python scripts/plt/generate_violin_plot.py --labels "GNN-RT" "VisNet-V2" --csv_files predictions\gnn-rt\pos-prediction_20251126-193532\MMF_GNN_pos_predictions.csv "D:\Projects\python\gnn-rt-1\predictions\visnet-v2\prediction_20251129-134345\test_predictions_restored.csv"

2. neg
cd d:\Projects\python\gnn-rt-1; python scripts/plt/generate_violin_plot.py --labels "GNN-RT" "VisNet-V2" --csv_files predictions/gnn-rt/neg-prediction_20251127-001446/MMF_GNN_neg_predictions.csv "predictions\visnet-v2\prediction_20251129-160839\test_predictions_restored.csv"
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import json
import datetime

def compute_metrics(preds, actuals, n_iterations=1000):
    """
    使用Bootstrap重采样方法计算R2和MAE值
    
    Args:
        preds (np.array): 预测值数组
        actuals (np.array): 实际值数组
        n_iterations (int): Bootstrap迭代次数
        
    Returns:
        tuple: (r2_values, mae_values) R2和MAE值列表
    """
    # 检查并处理NaN或无穷大值
    valid_mask = np.isfinite(preds) & np.isfinite(actuals)
    if not np.all(valid_mask):
        preds = preds[valid_mask]
        actuals = actuals[valid_mask]
        print(f"Warning: Filtered out {np.sum(~valid_mask)} invalid predictions")
    
    if len(preds) == 0 or len(actuals) == 0:
        print("Error: No valid predictions found")
        return [], []
    
    # 使用Bootstrap重采样方法计算R2和MAE值
    n_samples = len(preds)
    sample_size = min(100, n_samples // 10) if n_samples >= 10 else n_samples  # 每次采样的大小
    
    r2_values = []
    mae_values = []
    
    # 进行Bootstrap重采样
    for _ in range(n_iterations):
        # 随机采样（可重复）
        indices = np.random.choice(n_samples, size=sample_size, replace=True)
        sample_preds = preds[indices]
        sample_actuals = actuals[indices]
        
        if len(sample_preds) > 1:  # 至少需要2个点才能计算R2
            # 计算R2
            ss_res = np.sum((sample_actuals - sample_preds) ** 2)
            ss_tot = np.sum((sample_actuals - np.mean(sample_actuals)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            r2_values.append(r2)
            
            # 计算MAE
            mae = np.mean(np.abs(sample_preds - sample_actuals))
            mae_values.append(mae)
        elif len(sample_preds) == 1:  # 只有一个点的情况
            # 无法计算R2，设为0
            r2_values.append(0)
            # 计算MAE
            mae = np.abs(sample_preds[0] - sample_actuals[0])
            mae_values.append(mae)
    
    return r2_values, mae_values

def plot_violin_metrics_from_csv(csv_files, labels=None, output_dir=None):
    """
    根据预测结果CSV文件生成R2和MAE指标的小提琴图
    
    Args:
        csv_files (list): 预测结果CSV文件路径列表
        labels (list): 每个文件对应的标签列表
        output_dir (str): 输出目录路径，默认为第一个CSV文件所在目录
        
    Returns:
        str: 生成的小提琴图文件路径
    """
    if labels is None:
        # 从文件名生成默认标签
        labels = [os.path.splitext(os.path.basename(f))[0] for f in csv_files]
    
    # 创建带时间戳的输出目录
    if output_dir is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        output_dir = os.path.join('results', 'violin', f'violin_{timestamp}')
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 存储所有数据
    all_r2_values = []
    all_mae_values = []
    all_labels = []
    sample_counts = []
    
    # 处理每个CSV文件
    for csv_file, label in zip(csv_files, labels):
        # 读取CSV文件
        df = pd.read_csv(csv_file)
        
        # 提取预测值和实际值
        preds = df['Predicted'].values
        actuals = df['Actual'].values
        
        # 计算指标
        r2_values, mae_values = compute_metrics(preds, actuals)
        
        # 存储数据
        all_r2_values.append(r2_values)
        all_mae_values.append(mae_values)
        all_labels.append(label)
        sample_counts.append(len(preds))
    
    # 创建小提琴图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 绘制R2小提琴图
    if all_r2_values:
        # 为每个算法设置颜色
        colors = ['#66c2a5', '#fc8d52', '#8da0cb', '#fdae6b', '#ff7f00']
        
        # 绘制R2小提琴图
        vp1 = ax1.violinplot(all_r2_values, positions=range(1, len(all_r2_values) + 1), 
                            showmeans=True, showmedians=True)
        
        # 设置颜色
        for i, pc in enumerate(vp1['bodies']):
            pc.set_facecolor(colors[i % len(colors)])
            pc.set_alpha(0.7)
        
        vp1['cmeans'].set_color('black')
        vp1['cmedians'].set_color('black')
        
        ax1.set_title('R² Score Distribution')
        ax1.set_ylabel('R² Score')
        ax1.set_xticks(range(1, len(all_labels) + 1))
        ax1.set_xticklabels(all_labels)
        ax1.grid(True, alpha=0.3)
    
    # 绘制MAE小提琴图
    if all_mae_values:
        # 为每个算法设置颜色
        colors = ['#66c2a5', '#fc8d52', '#8da0cb', '#fdae6b', '#ff7f00']
        
        # 绘制MAE小提琴图
        vp2 = ax2.violinplot(all_mae_values, positions=range(1, len(all_mae_values) + 1), 
                            showmeans=True, showmedians=True)
        
        # 设置颜色
        for i, pc in enumerate(vp2['bodies']):
            pc.set_facecolor(colors[i % len(colors)])
            pc.set_alpha(0.7)
        
        vp2['cmeans'].set_color('black')
        vp2['cmedians'].set_color('black')
        
        ax2.set_title('MAE Distribution')
        ax2.set_ylabel('Mean Absolute Error')
        ax2.set_xticks(range(1, len(all_labels) + 1))
        ax2.set_xticklabels(all_labels)
        ax2.grid(True, alpha=0.3)
    
    # 添加样本数量信息
    items_text = ', '.join([f'{label}: {count}' for label, count in zip(all_labels, sample_counts)])
    plt.suptitle(f'R² and MAE Violin Plots Comparison\n{items_text}')
    plt.tight_layout()
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存小提琴图
    violin_filename = 'comparison_violin_metrics.png'
    violin_filepath = os.path.join(output_dir, violin_filename)
    plt.savefig(violin_filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存绘图数据到JSON文件
    plot_data = {
        'timestamp': datetime.datetime.now().isoformat(),
        'algorithms': all_labels,
        'sample_counts': sample_counts,
        'r2_values': {label: values for label, values in zip(all_labels, all_r2_values)},
        'mae_values': {label: values for label, values in zip(all_labels, all_mae_values)}
    }
    
    json_filename = 'violin_plot_data.json'
    json_filepath = os.path.join(output_dir, json_filename)
    with open(json_filepath, 'w') as f:
        json.dump(plot_data, f, indent=2)
    
    print(f"Violin plots saved to {violin_filepath}")
    print(f"Plot data saved to {json_filepath}")
    
    return violin_filepath

def main():
    parser = argparse.ArgumentParser(description='Generate violin plots of R2 and MAE metrics from prediction CSV file(s)')
    parser.add_argument('--csv_files', type=str, nargs='+', required=True, 
                        help='Path(s) to the prediction CSV file(s)')
    parser.add_argument('--labels', type=str, nargs='+', default=None,
                        help='Labels for each prediction file (default: filenames)')
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='Output directory for the violin plot (default: results/violin with timestamp)')
    
    args = parser.parse_args()
    
    # 生成小提琴图
    plot_violin_metrics_from_csv(args.csv_files, args.labels, args.output_dir)

if __name__ == '__main__':
    main()