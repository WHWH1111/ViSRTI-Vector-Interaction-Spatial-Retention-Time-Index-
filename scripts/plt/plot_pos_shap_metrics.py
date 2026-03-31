#!/usr/bin/env python3
"""
根据 pos-shap.csv 文件中的测试指标数据绘制柱状图和折线图

该脚本读取 pos-shap.csv 文件中的测试集指标数据（MAE_test, MSE_test, RMSE_test, R2_test），
并生成组合图表：柱状图显示各模型的指标值，折线图显示指标的变化趋势。
同时将绘图数据保存为JSON文件。

cd d:\Projects\python\gnn-rt-1; python scripts/plt/plot_pos_shap_metrics.py

# 旧版本
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import datetime


def plot_pos_shap_metrics(csv_file, output_dir=None):
    """
    根据CSV文件中的测试指标数据绘制柱状图和折线图
    
    Args:
        csv_file (str): 包含指标数据的CSV文件路径
        output_dir (str): 图表输出目录路径，默认为None（使用CSV文件所在目录）
    
    Returns:
        str: 生成的图表文件路径
    """
    # 读取CSV文件，使用制表符分隔符
    df = pd.read_csv(csv_file, sep='\t')
    
    # 提取模型名称和测试指标
    models = df['Title'].tolist()
    mae_test = df['MAE_test'].tolist()
    rmse_test = df['RMSE_test'].tolist()
    r2_test = df['R2_test'].tolist()
    
    # 如果没有指定输出目录，则使用results目录
    if output_dir is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        output_dir = os.path.join('results', 'pos_shap_metrics', f'pos_shap_{timestamp}')
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建图表
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # 设置柱状图参数
    x = np.arange(len(models))  # 模型索引
    width = 0.25  # 柱状图宽度
    
    # 绘制柱状图（三个指标居中分布）
    bars1 = ax1.bar(x - width, mae_test, width, label='MAE', color='#66c2a5', alpha=0.8)
    bars2 = ax1.bar(x, rmse_test, width, label='RMSE', color='#a6dba0', alpha=0.8)
    bars3 = ax1.bar(x + width, r2_test, width, label='R²', color='#8da0cb', alpha=0.8)
    
    # 在柱状图上添加数值标签
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    # 创建第二个y轴用于折线图
    ax2 = ax1.twinx()
    
    # 绘制折线图
    line1, = ax2.plot(x, mae_test, marker='o', color='#66c2a5', linewidth=2, label='MAE Trend')
    line2, = ax2.plot(x, rmse_test, marker='v', color='#a6dba0', linewidth=2, label='RMSE Trend')
    line3, = ax2.plot(x, r2_test, marker='^', color='#8da0cb', linewidth=2, label='R² Trend')
    
    # 设置图表标题和轴标签
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Metric Values (Bar Chart)')
    ax2.set_ylabel('Metric Values (Line Trend)')
    ax1.set_title('SHAP Test Metrics Comparison - Bar and Line Chart')
    
    # 设置x轴刻度
    ax1.set_xticks(x)
    # 处理过长的模型名称
    formatted_models = []
    for model in models:
        if len(model) > 30:
            formatted_models.append(model[:27] + '...')
        else:
            formatted_models.append(model)
            
    ax1.set_xticklabels(formatted_models, rotation=45, ha='right')
    
    # 添加网格
    ax1.grid(True, alpha=0.3)
    
    # 创建图例
    bars_legend = ax1.legend(loc='upper left', bbox_to_anchor=(0, 0.9))
    lines_legend = ax2.legend(loc='upper right', bbox_to_anchor=(1, 0.9))
    
    # 添加样本数量信息
    items_text = f'Items = {len(models)}'
    plt.figtext(0.99, 0.01, items_text, horizontalalignment='right', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    # 调整布局，为底部的长标签留出空间
    plt.tight_layout()
    
    # 保存图表
    output_file = os.path.join(output_dir, 'pos_shap_metrics_bar_line_chart.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存绘图数据到JSON文件
    plot_data = {
        'timestamp': datetime.datetime.now().isoformat(),
        'models': models,
        'metrics': {
            'MAE_test': mae_test,
            'RMSE_test': rmse_test,
            'R2_test': r2_test
        },
        'sample_count': len(models)
    }
    
    json_filename = 'pos_shap_metrics_data.json'
    json_filepath = os.path.join(output_dir, json_filename)
    with open(json_filepath, 'w') as f:
        json.dump(plot_data, f, indent=2)
    
    print(f"SHAP test metrics bar and line chart saved to: {output_file}")
    print(f"Plot data saved to: {json_filepath}")
    return output_file


def main():
    """主函数"""
    # 输入文件路径
    csv_file = r"data/bar-line/pos-shap.csv"
    # csv_file = r"data/bar-line/neg-shap.csv"
    
    # 检查文件是否存在
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' does not exist.")
        return
    
    # 生成图表
    plot_pos_shap_metrics(csv_file)


if __name__ == "__main__":
    main()