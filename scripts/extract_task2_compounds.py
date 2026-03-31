#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据 task-2.csv 中的化合物列表，从 norman 数据库中提取完整信息
"""

import pandas as pd
import os

def extract_compounds():
    # 定义文件路径
    task2_file = os.path.join('data', 'MMF-4', 'task-2.csv')
    norman_db_file = os.path.join('data', 'MMF-4', 'norman数据库.csv')
    output_file = os.path.join('data', 'MMF-4', 'task-2-full.csv')
    
    # 读取 task-2.csv 文件
    task2_df = pd.read_csv(task2_file)
    print(f"Task-2 文件中有 {len(task2_df)} 条记录")
    
    # 读取 norman 数据库文件
    norman_df = pd.read_csv(norman_db_file, low_memory=False)
    print(f"Norman 数据库中有 {len(norman_df)} 条记录")
    
    # 获取 task-2 中的 Norman_SusDat_ID 列表
    task2_ids = task2_df['Norman_SusDat_ID'].tolist()
    print(f"需要查找 {len(task2_ids)} 个化合物 ID")
    
    # 从 norman 数据库中筛选出对应的记录
    filtered_df = norman_df[norman_df['Norman_SusDat_ID'].isin(task2_ids)]
    print(f"找到 {len(filtered_df)} 条匹配记录")
    
    # 合并 task-2 中的中文名称信息
    merged_df = pd.merge(task2_df, filtered_df, on='Norman_SusDat_ID', how='left')
    
    # 保存结果到新文件
    merged_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"结果已保存到 {output_file}")
    
    # 显示基本信息
    print("\n提取的数据列:")
    for col in merged_df.columns:
        print(f"  - {col}")

if __name__ == "__main__":
    extract_compounds()