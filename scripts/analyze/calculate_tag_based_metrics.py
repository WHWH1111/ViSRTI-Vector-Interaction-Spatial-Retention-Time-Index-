#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据tags分类计算预测结果的评估指标

该脚本分别读取正离子和负离子模式下的merged_predictions_with_tags.csv文件，
按照不同的tag类别计算items数量、R2、MAE指标，并输出汇总结果。
"""

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
import os

def load_data(filepath):
    """
    加载预测数据
    
    Args:
        filepath (str): CSV文件路径
        
    Returns:
        DataFrame: 加载的数据
    """
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def calculate_metrics_for_tag(df, tag=None):
    """
    计算特定tag或总体的评估指标
    
    Args:
        df (DataFrame): 数据框
        tag (str, optional): 特定的tag，如果为None则计算总体指标
        
    Returns:
        dict: 包含items数量、R2和MAE的字典
    """
    # 如果指定了tag，则筛选包含该tag的数据
    if tag:
        filtered_df = df[df['Tags'].str.contains(tag, na=False)]
    else:
        filtered_df = df
    
    # 获取真实值和预测值
    y_true = filtered_df['Actual'].values
    y_pred = filtered_df['Predicted'].values
    
    # 检查是否有有效数据
    if len(y_true) == 0:
        return {
            'items': 0,
            'r2': np.nan,
            'mae': np.nan
        }
    
    # 计算指标
    items = len(y_true)
    try:
        r2 = r2_score(y_true, y_pred)
    except:
        r2 = np.nan
        
    try:
        mae = mean_absolute_error(y_true, y_pred)
    except:
        mae = np.nan
    
    return {
        'items': items,
        'r2': r2,
        'mae': mae
    }

def get_all_tags(df):
    """
    从数据中获取所有唯一的标签类别
    
    Args:
        df (DataFrame): 包含Tags列的数据框
        
    Returns:
        list: 排序后的唯一标签列表
    """
    all_tags = set()
    for tags_str in df['Tags']:
        if pd.notna(tags_str):  # 检查是否为NaN
            tags = tags_str.split(',')
            all_tags.update(tags)
    
    return sorted(list(all_tags))

def analyze_file(filepath, ion_mode):
    """
    分析单个文件的所有tag类别
    
    Args:
        filepath (str): CSV文件路径
        ion_mode (str): 离子模式 ('pos' 或 'neg')
        
    Returns:
        tuple: (results_dict, tag_categories)
    """
    print(f"Analyzing {ion_mode} ion mode: {filepath}")
    
    df = load_data(filepath)
    if df is None:
        return None, []
    
    # 动态获取所有标签类别
    tag_categories = get_all_tags(df)
    
    results = {}
    
    # 计算总体指标
    results['Overall'] = calculate_metrics_for_tag(df)
    
    # 计算各个tag类别的指标
    for tag in tag_categories:
        results[tag] = calculate_metrics_for_tag(df, tag)
    
    return results, tag_categories

def print_results(results, tag_categories, ion_mode):
    """
    打印分析结果
    
    Args:
        results (dict): 分析结果
        tag_categories (list): 标签类别列表
        ion_mode (str): 离子模式
    """
    print(f"\n=== {ion_mode.upper()} Ion Mode Results ===")
    print(f"{'Category':<25} {'Items':<10} {'R2':<12} {'MAE':<12}")
    print("-" * 60)
    
    # 输出总体结果
    overall = results['Overall']
    print(f"{'Overall':<25} {overall['items']:<10} {overall['r2']:<12.4f} {overall['mae']:<12.4f}")
    
    # 输出各类别结果
    for tag in tag_categories:
        metrics = results[tag]
        if metrics['items'] > 0:
            print(f"{tag:<25} {metrics['items']:<10} {metrics['r2']:<12.4f} {metrics['mae']:<12.4f}")
        else:
            print(f"{tag:<25} {metrics['items']:<10} {'nan':<12} {'nan':<12}")

def main():
    # 定义文件路径
    pos_file = "predictions/visnet-v2-5fold/pos/merged_predictions_with_tags.csv"
    neg_file = "predictions/visnet-v2-5fold/neg/merged_predictions_with_tags.csv"
    
    # 检查文件是否存在
    if not os.path.exists(pos_file):
        print(f"Error: Positive ion mode file not found: {pos_file}")
        return
    
    if not os.path.exists(neg_file):
        print(f"Error: Negative ion mode file not found: {neg_file}")
        return
    
    # 分析正离子模式
    pos_results, pos_tags = analyze_file(pos_file, 'pos')
    if pos_results:
        print_results(pos_results, pos_tags, 'pos')
    
    # 分析负离子模式
    neg_results, neg_tags = analyze_file(neg_file, 'neg')
    if neg_results:
        print_results(neg_results, neg_tags, 'neg')

if __name__ == "__main__":
    main()