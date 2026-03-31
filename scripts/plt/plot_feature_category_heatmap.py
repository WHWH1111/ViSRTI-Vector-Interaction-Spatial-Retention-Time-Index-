#!/usr/bin/env python3
"""
生成特征-类别关联热力图，展示不同类别污染物其RT预测所依赖的关键特征差异。

X轴：特征（Features）
Y轴：污染物类别（Categories） 
颜色（Z轴）：单元格颜色表示归一化后的平均|SHAP value|
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

from sklearn import base

def load_shap_data(base_dir):
    """
    从指定目录加载所有分类的SHAP分析结果
    
    Args:
        base_dir (str): 包含所有分类SHAP结果的基目录路径
        
    Returns:
        dict: 每个分类的特征重要性数据
    """
    shap_data = {}
    
    # 遍历所有分类目录
    for category_dir in Path(base_dir).iterdir():
        if category_dir.is_dir():
            category_name = category_dir.name
            
            # 查找最新的SHAP分析结果目录
            shap_analysis_dirs = list(category_dir.iterdir())
            if not shap_analysis_dirs:
                continue
                
            # 假设只有一个SHAP分析目录或者使用最新的
            latest_shap_dir = sorted(shap_analysis_dirs)[-1]
            
            # SHAP结果文件路径
            shap_result_file = latest_shap_dir / "shap_analysis_results.json"
            
            if shap_result_file.exists():
                with open(shap_result_file, 'r') as f:
                    data = json.load(f)
                    
                shap_data[category_name] = {
                    'feature_names': data['feature_names'],
                    'mean_abs_shap': data['mean_abs_shap']
                }
                
    return shap_data

def normalize_shap_by_category(shap_data):
    """
    对每个分类的SHAP值进行归一化处理
    
    Args:
        shap_data (dict): 原始SHAP数据
        
    Returns:
        dict: 归一化后的SHAP数据
    """
    normalized_data = {}
    
    for category, data in shap_data.items():
        shap_values = np.array(data['mean_abs_shap'])
        
        # 归一化：每个分类内的特征重要性进行归一化 (0-1)
        min_val = shap_values.min()
        max_val = shap_values.max()
        
        # 避免除零错误
        if max_val - min_val != 0:
            normalized_shap = (shap_values - min_val) / (max_val - min_val)
        else:
            normalized_shap = np.zeros_like(shap_values)
            
        normalized_data[category] = {
            'feature_names': data['feature_names'],
            'mean_abs_shap': normalized_shap.tolist()
        }
        
    return normalized_data

def create_heatmap_data(normalized_data):
    """
    构建用于热力图的数据矩阵
    
    Args:
        normalized_data (dict): 归一化后的SHAP数据
        
    Returns:
        pd.DataFrame: 用于绘制热力图的数据框
    """
    # 定义特征顺序，按照data_preprocessor.py中的定义
    feature_order = [
        'Monoiso_Mass',
        'LogKow',  # 对应logKow/Exp_logKow
        'LogP',    # 对应alogp/xlogp
        'Tetrahymena_pyriformis_toxicity',
        'Daphnia_toxicity',
        'Algae_toxicity',
        'Pimephales_promelas_toxicity'
    ]
    
    # 收集所有特征名称
    all_features = set()
    for data in normalized_data.values():
        all_features.update(data['feature_names'])
    
    # 按照预定义顺序排列特征，未在列表中的特征放在最后并按字母顺序排列
    ordered_features = []
    for feature in feature_order:
        if feature in all_features:
            ordered_features.append(feature)
            
    # 添加剩余特征
    remaining_features = sorted(list(all_features - set(ordered_features)))
    ordered_features.extend(remaining_features)
    
    # 创建数据矩阵
    categories = list(normalized_data.keys())
    heatmap_data = []
    
    for category in categories:
        data = normalized_data[category]
        feature_dict = dict(zip(data['feature_names'], data['mean_abs_shap']))
        
        # 为当前分类创建一行数据，按照指定顺序排列
        row = [feature_dict.get(feature, 0.0) for feature in ordered_features]
        heatmap_data.append(row)
    
    # 创建DataFrame
    df = pd.DataFrame(heatmap_data, index=categories, columns=ordered_features)
    return df

def plot_feature_category_heatmap(df, output_file=None):
    """
    绘制特征-类别关联热力图
    
    Args:
        df (pd.DataFrame): 用于绘制热力图的数据框
        output_file (str): 输出文件路径
    """
    # 设置图形大小
    plt.figure(figsize=(12, 8))
    
    # 绘制热力图
    ax = sns.heatmap(df, 
                     annot=True, 
                     fmt='.2f', 
                     cmap='YlOrRd', 
                     cbar_kws={'label': 'Normalized SHAP Value'})
    
    # 设置标签
    plt.title('Feature-Category Association Heatmap\n(Normalized SHAP Values by Category)')
    plt.xlabel('Features')
    plt.ylabel('Compound Categories')
    
    # 旋转x轴标签以提高可读性
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存或显示图形
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()

def main():
    """主函数"""
    # 基础目录
    # base_dir = "results/shap/pos-3-type-shap"
    base_dir = r"results\shap\neg-3-type-shap"
    
    # 检查目录是否存在
    if not os.path.exists(base_dir):
        print(f"Error: Directory '{base_dir}' does not exist.")
        return
    
    # 加载SHAP数据
    print("Loading SHAP data...")
    shap_data = load_shap_data(base_dir)
    
    if not shap_data:
        print("Error: No SHAP data found.")
        return
    
    print(f"Loaded data for {len(shap_data)} categories.")
    
    # 归一化处理
    print("Normalizing SHAP values by category...")
    normalized_data = normalize_shap_by_category(shap_data)
    
    # 构建热力图数据
    print("Creating heatmap data...")
    df = create_heatmap_data(normalized_data)
    
    # 输出数据摘要
    print("\nHeatmap data summary:")
    print(df)
    
    # 创建带时间戳的输出目录，符合项目规范
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join("results", "feature_category_heatmap", f"feature_category_heatmap_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成输出文件路径
    heatmap_file = os.path.join(output_dir, f"feature_category_heatmap_{timestamp}.png")
    
    # 绘制并保存热力图
    print("Plotting heatmap...")
    plot_feature_category_heatmap(df, heatmap_file)
    
    # 保存数据到CSV文件
    csv_file = os.path.join(output_dir, f"feature_category_data_{timestamp}.csv")
    df.to_csv(csv_file)
    print(f"Data saved to: {csv_file}")
    print(f"Heatmap saved to: {heatmap_file}")

if __name__ == "__main__":
    main()