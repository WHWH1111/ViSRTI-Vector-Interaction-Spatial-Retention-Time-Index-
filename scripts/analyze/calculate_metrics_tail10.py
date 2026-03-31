#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算日志文件中最后10行的评估指标均值和标准误差
"""

import os
import numpy as np


def calculate_metrics_from_log(file_path):
    """
    从日志文件中提取最后10行的测试指标并计算均值和标准误差
    
    参数:
    file_path (str): 日志文件路径
    
    返回:
    dict: 包含各项指标均值和标准误差的字典
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # 跳过标题行
    data_lines = lines[1:]
    
    # 获取最后10行
    last_ten_lines = data_lines[-10:]
    
    # 提取指标:
    # MAE_test (第12列, 索引11)
    # MSE_test (第13列, 索引12) 
    # R2_test (第14列, 索引13)
    # PCC_test (第15列, 索引14)
    mae_test_values = []
    mse_test_values = []
    r2_test_values = []
    pcc_test_values = []
    
    for line in last_ten_lines:
        values = line.strip().split('\t')
        mae_test_values.append(float(values[11]))
        mse_test_values.append(float(values[12]))
        r2_test_values.append(float(values[13]))
        pcc_test_values.append(float(values[14]))
    
    # 计算均值
    mean_mae_test = np.mean(mae_test_values)
    mean_mse_test = np.mean(mse_test_values)
    mean_r2_test = np.mean(r2_test_values)
    mean_pcc_test = np.mean(pcc_test_values)
    
    # 计算标准误差 (标准差 / sqrt(n))
    se_mae_test = np.std(mae_test_values, ddof=1) / np.sqrt(len(mae_test_values))
    se_mse_test = np.std(mse_test_values, ddof=1) / np.sqrt(len(mse_test_values))
    se_r2_test = np.std(r2_test_values, ddof=1) / np.sqrt(len(r2_test_values))
    se_pcc_test = np.std(pcc_test_values, ddof=1) / np.sqrt(len(pcc_test_values))
    
    return {
        'mae_test': (mean_mae_test, se_mae_test),
        'mse_test': (mean_mse_test, se_mse_test),
        'r2_test': (mean_r2_test, se_r2_test),
        'pcc_test': (mean_pcc_test, se_pcc_test)
    }


def main():
    # 定义日志目录
    # log_dir = 'log/pos'
    log_dir = 'log/pos-mask'
    # log_dir = 'log/neg'
    # log_dir = 'log/neg-mask'
    
    print("日志文件最后10行指标统计结果:")
    print("=" * 50)
    
    # 处理每个日志文件
    for filename in os.listdir(log_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(log_dir, filename)
            results = calculate_metrics_from_log(file_path)
            
            print(f"\n文件: {filename}")
            print("-" * 30)
            print(f"{'指标':<12} {'均值':<12} {'标准误差':<12}")
            print("-" * 30)
            for metric, (mean, se) in results.items():
                print(f"{metric:<12} {mean:<12.6f} {se:<12.6f}")


if __name__ == "__main__":
    main()