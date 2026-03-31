#!/usr/bin/env python3
"""
根据 pos.csv 文件中的测试指标数据绘制柱状图和折线图

该脚本读取 pos.csv 文件中的测试集指标数据（MAE_test, MSE_test, R2_test, PCC_test），
并将MSE转换为RMSE，然后生成组合图表：
- 左侧Y轴显示MAE和RMSE的柱状图和趋势线
- 右侧Y轴显示R2和PCC的柱状图和趋势线
同时将绘图数据保存为JSON文件。

cd d:\Projects\python\gnn-rt-1; python scripts/plt/plot_test_metrics_bar_line.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import datetime
import math

# 设置支持中文的字体
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False

def plot_test_metrics_bar_line(csv_file, output_dir=None):
    """
    根据CSV文件中的测试指标数据绘制柱状图和折线图
    
    Args:
        csv_file (str): 包含指标数据的CSV文件路径
        output_dir (str): 图表输出目录路径，默认为None（使用CSV文件所在目录）
    
    Returns:
        str: 生成的图表文件路径
    """
    # 读取CSV文件，使用逗号分隔符
    df = pd.read_csv(csv_file)
    
    # 提取模型名称和测试指标（只取平均值部分）
    models = df['Model'].tolist()
    
    # 提取均值和标准差
    mae_mean = [float(str(val).split(' ± ')[0]) for val in df['MAE_test'].tolist()]
    mae_std = [float(str(val).split(' ± ')[1]) if ' ± ' in str(val) else 0 for val in df['MAE_test'].tolist()]
    
    mse_mean = [float(str(val).split(' ± ')[0]) for val in df['MSE_test'].tolist()]
    mse_std = [float(str(val).split(' ± ')[1]) if ' ± ' in str(val) else 0 for val in df['MSE_test'].tolist()]
    
    # 将MSE转换为RMSE
    rmse_mean = [math.sqrt(mse) for mse in mse_mean]
    rmse_std = [math.sqrt(mse) if mse >= 0 else 0 for mse in mse_std]
    
    r2_mean = [float(str(val).split(' ± ')[0]) for val in df['R2_test'].tolist()]
    r2_std = [float(str(val).split(' ± ')[1]) if ' ± ' in str(val) else 0 for val in df['R2_test'].tolist()]
    
    pcc_mean = [float(str(val).split(' ± ')[0]) for val in df['PCC_test'].tolist()]
    pcc_std = [float(str(val).split(' ± ')[1]) if ' ± ' in str(val) else 0 for val in df['PCC_test'].tolist()]
    
    # 获取标准化参数（每行使用自己的props_std值）
    props_std = df['props_std'].tolist()  # 获取所有行的props_std值
    
    # 对MAE和RMSE进行反标准化（均值和标准差都需要处理）
    # 对于误差类指标，反标准化公式为：原始值 = 标准化值 * std
    mae_mean_original = [mae * std for mae, std in zip(mae_mean, props_std)]
    mae_std_original = [std_mae * std for std_mae, std in zip(mae_std, props_std)]
    
    rmse_mean_original = [rmse * std for rmse, std in zip(rmse_mean, props_std)]
    rmse_std_original = [std_rmse * std for std_rmse, std in zip(rmse_std, props_std)]
    
    # 如果没有指定输出目录，则使用results目录
    if output_dir is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        output_dir = os.path.join('results', 'test_metrics_bar_line', f'barline_{timestamp}')
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建图表
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # 设置柱状图参数
    x = np.arange(len(models))  # 模型索引
    width = 0.2  # 柱状图宽度
    
    # 绘制柱状图（使用反标准化后的MAE和RMSE均值在左侧Y轴）
    bars1 = ax1.bar(x - 1.5*width, mae_mean_original, width, label='MAE Mean', color='#66c2a5', alpha=0.8)
    bars2 = ax1.bar(x - 0.5*width, rmse_mean_original, width, label='RMSE Mean', color='#fc8d62', alpha=0.8)
    
    # 添加置信区间区域（均值±标准差）
    ax1.fill_between(x, [m - s for m, s in zip(mae_mean_original, mae_std_original)], 
                     [m + s for m, s in zip(mae_mean_original, mae_std_original)], 
                     color='#66c2a5', alpha=0.2)
    ax1.fill_between(x, [m - s for m, s in zip(rmse_mean_original, rmse_std_original)], 
                     [m + s for m, s in zip(rmse_mean_original, rmse_std_original)], 
                     color='#fc8d62', alpha=0.2)
    
    # 在柱状图上添加数值标签
    def add_value_labels_ax1(bars, values):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.annotate(f'{value:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 12),  # 12 points vertical offset to make room for std label
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=7)
    
    # 在柱状图上添加标准差标签
    def add_std_labels_ax1(x_positions, means, stds, color):
        for x, mean, std in zip(x_positions, means, stds):
            ax1.annotate(f'±{std:.4f}',
                        xy=(x, mean),
                        xytext=(0, 0),  # Place std label just above the bar top
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=6, color='black')
    
    add_value_labels_ax1(bars1, mae_mean_original)
    add_value_labels_ax1(bars2, rmse_mean_original)
    add_std_labels_ax1(x - 1.5*width, mae_mean_original, mae_std_original, '#66c2a5')
    add_std_labels_ax1(x - 0.5*width, rmse_mean_original, rmse_std_original, '#fc8d62')

    
    # 创建第二个y轴用于R2和PCC
    ax2 = ax1.twinx()
    # 设置右侧Y轴范围为0-1
    ax2.set_ylim(0, 1)
    
    # 绘制R2和PCC柱状图（在右侧Y轴）
    bars3 = ax2.bar(x + 0.5*width, r2_mean, width, label='R² Mean', color='#8da0cb', alpha=0.8)
    bars4 = ax2.bar(x + 1.5*width, pcc_mean, width, label='PCC Mean', color='#e78ac3', alpha=0.8)
    
    # 添加置信区间区域（均值±标准差）
    ax2.fill_between(x, [max(0, m - s) for m, s in zip(r2_mean, r2_std)], 
                     [min(1, m + s) for m, s in zip(r2_mean, r2_std)], 
                     color='#8da0cb', alpha=0.2)
    ax2.fill_between(x, [max(0, m - s) for m, s in zip(pcc_mean, pcc_std)], 
                     [min(1, m + s) for m, s in zip(pcc_mean, pcc_std)], 
                     color='#e78ac3', alpha=0.2)
    # 为右侧轴添加数值标签
    def add_value_labels_ax2(bars, values):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.annotate(f'{value:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 12),  # 12 points vertical offset to make room for std label
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=7)
    
    # 在柱状图上添加标准差标签
    def add_std_labels_ax2(x_positions, means, stds, color):
        for x, mean, std in zip(x_positions, means, stds):
            ax2.annotate(f'±{std:.4f}',
                        xy=(x, mean),
                        xytext=(0, 0),  # Place std label just above the bar top
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=6, color='black')
    
    add_value_labels_ax2(bars3, r2_mean)
    add_value_labels_ax2(bars4, pcc_mean)
    add_std_labels_ax2(x + 0.5*width, r2_mean, r2_std, '#8da0cb')
    add_std_labels_ax2(x + 1.5*width, pcc_mean, pcc_std, '#e78ac3')
    
    # 绘制折线图（MAE和RMSE的趋势线在左侧Y轴，R2和PCC的趋势线在右侧Y轴）
    line1, = ax1.plot(x, mae_mean_original, marker='o', color='#66c2a5', linewidth=2, linestyle='-', label='MAE Mean Trend')
    line2, = ax1.plot(x, rmse_mean_original, marker='s', color='#fc8d62', linewidth=2, linestyle='-', label='RMSE Mean Trend')
    line3, = ax2.plot(x, r2_mean, marker='^', color='#8da0cb', linewidth=2, linestyle='-', label='R² Mean Trend')
    line4, = ax2.plot(x, pcc_mean, marker='d', color='#e78ac3', linewidth=2, linestyle='-', label='PCC Mean Trend')
    
    # 绘制标准差的误差条（分别对应左右两侧的Y轴）
    # 使用与柱状图相同颜色系但较浅的颜色表示标准差，增强视觉关联性
    ax1.errorbar(x - 1.5*width, mae_mean_original, yerr=mae_std_original, fmt='none', 
                 ecolor='#66c2a5', capsize=5, alpha=0.7, elinewidth=1.5, capthick=1.5)
    ax1.errorbar(x - 0.5*width, rmse_mean_original, yerr=rmse_std_original, fmt='none', 
                 ecolor='#fc8d62', capsize=5, alpha=0.7, elinewidth=1.5, capthick=1.5)
    ax2.errorbar(x + 0.5*width, r2_mean, yerr=r2_std, fmt='none', 
                 ecolor='#8da0cb', capsize=5, alpha=0.7, elinewidth=1.5, capthick=1.5)
    ax2.errorbar(x + 1.5*width, pcc_mean, yerr=pcc_std, fmt='none', 
                 ecolor='#e78ac3', capsize=5, alpha=0.7, elinewidth=1.5, capthick=1.5)
    
    # 设置图表标题和轴标签
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Metric Values (MAE & RMSE)')
    ax2.set_ylabel('Metric Values (R² & PCC)')
    ax1.set_title('Test Metrics Comparison - Bar and Line Chart')
    
    # 设置x轴刻度
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    
    # 添加网格
    ax1.grid(True, alpha=0.3)
    
    # 创建图例
    # 收集柱状图的句柄和标签
    bars_handles = [bars1, bars2, bars3, bars4]
    bars_labels = ['MAE Mean', 'RMSE Mean', 'R² Mean', 'PCC Mean']
    
    # 创建柱状图图例（左下角）
    ax1.legend(bars_handles, bars_labels, loc='lower left')
    
    # 创建折线图图例（右下角）
    ax2.legend([line1, line2, line3, line4], 
               ['MAE Trend', 'RMSE Trend', 'R² Trend', 'PCC Trend'],
               loc='lower right')
    
    # 添加标准差说明文本
    plt.figtext(0.02, 0.01, 'Shaded areas represent mean ± standard deviation', 
                fontsize=10, style='italic',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
    
    # 添加样本数量信息
    items_text = f'Items = {len(models)}'
    plt.figtext(0.99, 0.01, items_text, horizontalalignment='right', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    output_file = os.path.join(output_dir, 'test_metrics_bar_line_chart.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存绘图数据到JSON文件
    plot_data = {
        'timestamp': datetime.datetime.now().isoformat(),
        'models': models,
        'metrics': {
            'MAE_mean': mae_mean_original,
            'MAE_std': mae_std_original,
            'RMSE_mean': rmse_mean_original,
            'RMSE_std': rmse_std_original,
            'R2_mean': r2_mean,
            'R2_std': r2_std,
            'PCC_mean': pcc_mean,
            'PCC_std': pcc_std
        },
        'sample_count': len(models),
        'standardization': {
            'props_std': props_std,
            'original_MAE_mean': mae_mean,
            'original_RMSE_mean': rmse_mean,
            'original_MAE_std': mae_std,
            'original_RMSE_std': rmse_std
        }
    }
    
    json_filename = 'test_metrics_data.json'
    json_filepath = os.path.join(output_dir, json_filename)
    with open(json_filepath, 'w') as f:
        json.dump(plot_data, f, indent=2)
    
    print(f"Test metrics bar and line chart saved to: {output_file}")
    print(f"Plot data saved to: {json_filepath}")
    return output_file

def main():
    """主函数"""
    # 输入文件路径
    # csv_file = r"data/bar-line/pos.csv"
    csv_file = r"data/bar-line/pos-mask.csv"
    # csv_file = r"data/bar-line/neg.csv"
    # csv_file = r"data/bar-line/neg-mask.csv"
    
    # 检查文件是否存在
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' does not exist.")
        return
    
    # 生成图表
    plot_test_metrics_bar_line(csv_file)

if __name__ == "__main__":
    main()