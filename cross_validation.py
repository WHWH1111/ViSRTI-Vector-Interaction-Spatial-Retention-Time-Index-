#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VisNetV2模型的5折交叉验证

该脚本对VisNetV2模型执行5折交叉验证。
"""

import os
import sys
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from collections import Counter
import argparse
import json

# 将项目根目录添加到路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def create_fold_directories(base_path, dataname, n_folds=5):
    """
    为每个折创建目录
    
    参数:
        base_path (str): 数据集的基础路径
        dataname (str): 数据集名称
        n_folds (int): 交叉验证的折数
        
    返回:
        list: 折目录路径列表
    """
    fold_dirs = []
    for fold in range(n_folds):
        fold_dir = os.path.join(base_path, f"{dataname}_5", f"fold_{fold+1}")
        os.makedirs(fold_dir, exist_ok=True)
        fold_dirs.append(fold_dir)
    return fold_dirs

def assign_compound_categories(tags_df):
    """
    根据标签为化合物分配类别用于分层
    
    参数:
        tags_df (DataFrame): 包含SMILES和Tags列的DataFrame
        
    返回:
        Series: 包含化合物类别的Series
    """
    # 定义分类的优先级顺序
    hydrophobicity_priority = ['High_hydrophobic', 'Medium_hydrophobic', 'Low_hydrophobic', 'Hydrophilic']
    element_priority = ['Halogen_containing', 'Sulfur_containing', 'Nitrogen_containing', 'Oxygen_containing']
    
    categories = []
    for idx, row in tags_df.iterrows():
        tags = row['Tags'].split(',')
        
        # 确定疏水性类别
        hydro_category = 'Other'  # 默认值
        for h in hydrophobicity_priority:
            if h in tags:
                hydro_category = h
                break
                
        # 确定元素类别（最具特色的元素）
        element_category = 'Other'  # 默认值
        for e in element_priority:
            if e in tags:
                element_category = e
                break
                
        # 组合类别
        category = f"{hydro_category}_{element_category}"
        categories.append(category)
        
    return pd.Series(categories, index=tags_df.index)

def split_data_for_cross_validation(data_file, tags_file=None, n_folds=5, random_state=1234):
    """
    将数据拆分为n折用于交叉验证
    
    参数:
        data_file (str): 数据文件的路径
        tags_file (str): 用于分层的标签文件路径
        n_folds (int): 折数
        random_state (int): 用于重现性的随机种子
        
    返回:
        list: 每个折的训练/测试拆分列表
    """
    # 读取数据
    df = pd.read_csv(data_file)
    
    # 删除目标列中包含缺失值的行
    target_col_pos = 'Pred_RTI_Positive_ESI'
    target_col_neg = 'Pred_RTI_Negative_ESI'
    
    if target_col_pos in df.columns:
        df_clean = df[['SMILES', target_col_pos]].dropna()
        target_col = target_col_pos
    elif target_col_neg in df.columns:
        df_clean = df[['SMILES', target_col_neg]].dropna()
        target_col = target_col_neg
    else:
        raise ValueError(f"Neither {target_col_pos} nor {target_col_neg} found in the dataset")
    
    # 打乱数据
    df_clean = df_clean.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # 如果提供了标签文件，则使用分层拆分
    if tags_file and os.path.exists(tags_file):
        print("基于化合物标签使用分层拆分...")
        # 加载标签
        tags_df = pd.read_csv(tags_file)
        
        # 将数据与标签合并
        merged_df = df_clean.merge(tags_df, on='SMILES', how='inner')
        print(f"在总共{len(df_clean)}个化合物中，匹配到{len(merged_df)}个带有标签的化合物")
        
        # 为分层分配类别
        categories = assign_compound_categories(merged_df)
        merged_df['category'] = categories
        
        # 打印类别分布
        print("类别分布：")
        category_counts = Counter(categories)
        for category, count in sorted(category_counts.items()):
            print(f"  {category}: {count}")
        
        # 使用分层k折
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        
        folds = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(merged_df, merged_df['category'])):
            train_data = merged_df.iloc[train_idx][['SMILES', target_col]]
            test_data = merged_df.iloc[test_idx][['SMILES', target_col]]
            folds.append((train_data, test_data, target_col))
    else:
        print("使用常规K-Fold拆分...")
        # 创建KFold拆分
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        
        folds = []
        for fold, (train_idx, test_idx) in enumerate(kf.split(df_clean)):
            train_data = df_clean.iloc[train_idx]
            test_data = df_clean.iloc[test_idx]
            folds.append((train_data, test_data, target_col))
    
    return folds

def save_fold_data(folds, fold_dirs, dataname):
    """
    将折数据保存到各自的目录
    
    参数:
        folds (list): 每个折的训练/测试拆分列表
        fold_dirs (list): 折目录路径列表
        dataname (str): 数据集名称
    """
    for i, (train_data, test_data, target_col) in enumerate(folds):
        fold_dir = fold_dirs[i]
        
        # 保存训练和测试数据
        train_file = os.path.join(fold_dir, f'fold_{i+1}_train_set.txt')
        test_file = os.path.join(fold_dir, f'fold_{i+1}_test_set.txt')
        
        train_data.to_csv(train_file, sep='\t', index=False, header=False)
        test_data.to_csv(test_file, sep='\t', index=False, header=False)
        
        print(f"Fold {i+1}: {len(train_data)} training samples, {len(test_data)} test samples")


def run_cross_validation(dataname='MMF_GNN_pos', n_folds=5):
    """
    运行交叉验证
    
    参数:
        dataname (str): 数据集名称
        n_folds (int): 折数
    """
    base_path = './data/MMF-3'
    data_file = os.path.join(base_path, f'{dataname}.csv')
    original_cache_dir = os.path.join(base_path, dataname)
    
    # 根据数据集名称确定标签文件
    if 'pos' in dataname.lower():
        tags_file = os.path.join(base_path, 'tags', 'pos_compound_tags.csv')
    elif 'neg' in dataname.lower():
        tags_file = os.path.join(base_path, 'tags', 'neg_compound_tags.csv')
    else:
        tags_file = None
    
    # 创建折目录
    print("正在创建折目录...")
    fold_dirs = create_fold_directories(base_path, dataname, n_folds)
    
    # 将数据拆分为折
    print("正在将数据拆分为折...")
    folds = split_data_for_cross_validation(data_file, tags_file, n_folds)
    
    # 保存折数据
    print("正在保存折数据...")
    save_fold_data(folds, fold_dirs, dataname)
    
def main():
    parser = argparse.ArgumentParser(description='为VisNetV2设置5折交叉验证')
    parser.add_argument('--dataname', type=str, default='MMF_GNN_pos',
                        help='数据集名称 (默认: MMF_GNN_pos)')
    parser.add_argument('--folds', type=int, default=5,
                        help='折数 (默认: 5)')
    
    args = parser.parse_args()
    
    print(f"为{args.dataname}设置{args.folds}-折交叉验证")
    run_cross_validation(args.dataname, args.folds)

if __name__ == "__main__":
    main()