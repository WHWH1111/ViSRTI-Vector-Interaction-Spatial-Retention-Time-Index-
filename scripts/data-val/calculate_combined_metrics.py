#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算多个预测文件的联合评估指标

该脚本读取多个预测文件，合并它们的数据，然后计算联合的MAE、R2、PCC、RMSE指标，
并以Markdown表格格式输出结果。
"""

import numpy as np
import os
import sys
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def read_prediction_file(file_path):
    """
    读取预测文件并返回预测值和真实值
    
    Args:
        file_path (str): 预测文件路径
        
    Returns:
        tuple: (smiles_list, true_values, pred_values)
    """
    smiles_list = []
    true_values = []
    pred_values = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    # 跳过标题行
    for line in lines[1:]:
        if line.strip():
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                smiles = parts[0]
                true_val = float(parts[1])
                pred_val = float(parts[2])
                
                smiles_list.append(smiles)
                true_values.append(true_val)
                pred_values.append(pred_val)
    
    return smiles_list, true_values, pred_values


def calculate_metrics(true_values, pred_values):
    """
    计算评估指标
    
    Args:
        true_values (list): 真实值列表
        pred_values (list): 预测值列表
        
    Returns:
        dict: 包含各种评估指标的字典
    """
    # 转换为numpy数组
    y_true = np.array(true_values)
    y_pred = np.array(pred_values)
    
    # 检查并处理NaN或无穷大值
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.all(valid_mask):
        print(f"Warning: {np.sum(~valid_mask)} values contain NaN or Inf, filtering them out")
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
    
    # 确保有足够的有效数据进行计算
    if len(y_true) == 0 or len(y_pred) == 0:
        print("Error: No valid predictions after filtering NaN/Inf values")
        return {
            "mae": float('nan'),
            "mse": float('nan'),
            "rmse": float('nan'),
            "r2": float('nan'),
            "pcc": float('nan')
        }
    
    # 计算各项指标
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # 只有当样本数大于1时才计算R²和PCC，避免警告
    if len(y_true) > 1:
        r2 = r2_score(y_true, y_pred)
        # 计算PCC (Pearson Correlation Coefficient)
        if np.std(y_pred) != 0 and np.std(y_true) != 0:
            pcc = np.corrcoef(y_true, y_pred)[0, 1]
        else:
            pcc = 0.0
    else:
        r2 = 0.0
        pcc = 0.0
        
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "pcc": pcc
    }


def process_single_file(file_path):
    """
    处理单个文件并返回其指标
    
    Args:
        file_path (str): 文件路径
        
    Returns:
        dict: 包含文件名和指标的字典
    """
    if not os.path.exists(file_path):
        print(f"Error: Prediction file not found: {file_path}")
        return None
    
    print(f"Reading prediction file: {file_path}")
    smiles, true_vals, pred_vals = read_prediction_file(file_path)
    print(f"Loaded {len(true_vals)} predictions from {os.path.basename(file_path)}")
    
    metrics = calculate_metrics(true_vals, pred_vals)
    
    return {
        "filename": os.path.basename(file_path),
        "count": len(true_vals),
        "metrics": metrics
    }


def process_combined_files(file_paths):
    """
    处理多个文件并计算联合指标
    
    Args:
        file_paths (list): 文件路径列表
        
    Returns:
        dict: 包含文件名列表和联合指标的字典
    """
    all_true = []
    all_pred = []
    all_smiles = []
    file_details = []
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Error: Prediction file not found: {file_path}")
            continue
            
        print(f"Reading prediction file: {file_path}")
        smiles, true_vals, pred_vals = read_prediction_file(file_path)
        print(f"Loaded {len(true_vals)} predictions from {os.path.basename(file_path)}")
        
        all_true.extend(true_vals)
        all_pred.extend(pred_vals)
        all_smiles.extend(smiles)
        
        # 保存单个文件的详情
        metrics = calculate_metrics(true_vals, pred_vals)
        file_details.append({
            "filename": os.path.basename(file_path),
            "count": len(true_vals),
            "metrics": metrics
        })
    
    print(f"\nCombined dataset contains {len(all_true)} predictions")
    
    # 计算联合指标
    print("\nCalculating combined metrics...")
    combined_metrics = calculate_metrics(all_true, all_pred)
    
    return {
        "files": file_details,
        "combined": {
            "filename": "Combined",
            "count": len(all_true),
            "metrics": combined_metrics
        }
    }


def format_as_markdown(results):
    """
    将结果格式化为Markdown表格
    
    Args:
        results (dict): 处理结果
        
    Returns:
        str: Markdown表格字符串
    """
    # 表头
    markdown = "| Dataset | Count | MAE | MSE | RMSE | R² | PCC |\n"
    markdown += "|---------|-------|-----|-----|------|----|-----|\n"
    
    # 单个文件结果
    for file_result in results["files"]:
        filename = file_result["filename"]
        count = file_result["count"]
        metrics = file_result["metrics"]
        
        markdown += f"| {filename} | {count} | {metrics['mae']:.4f} | {metrics['mse']:.4f} | {metrics['rmse']:.4f} | {metrics['r2']:.4f} | {metrics['pcc']:.4f} |\n"
    
    # 联合结果
    combined = results["combined"]
    markdown += f"| **{combined['filename']}** | **{combined['count']}** | **{combined['metrics']['mae']:.4f}** | **{combined['metrics']['mse']:.4f}** | **{combined['metrics']['rmse']:.4f}** | **{combined['metrics']['r2']:.4f}** | **{combined['metrics']['pcc']:.4f}** |\n"
    
    return markdown


def main():
    # 如果没有提供参数，则使用默认文件
    if len(sys.argv) == 1:
        # 默认文件路径
        default_dir = r"D:\Projects\python\gnn-rt-1\log\data-2\neg-3-mask(logkow)-train_20251113-112643_dim48_layerH6_layerO6_batch64_lr0.0001_iter150"
        file_paths = [
            # os.path.join(default_dir, "train_prediction_train.txt"),
            os.path.join(default_dir, "train_prediction_dev.txt"),
            os.path.join(default_dir, "train_prediction_test.txt")
        ]
        print("No input files provided. Using default files:")
        for path in file_paths:
            print(f"  {path}")
    else:
        # 使用命令行参数作为文件路径
        file_paths = sys.argv[1:]
    
    # 检查至少有一个文件
    if len(file_paths) == 0:
        print("Error: No files provided.")
        print("Usage: python calculate_combined_metrics.py [file1] [file2] ...")
        return
    
    # 处理文件
    results = process_combined_files(file_paths)
    
    if not results["files"]:
        print("Error: No valid files processed.")
        return
    
    # 输出Markdown表格
    print("\n" + "="*60)
    print("EVALUATION METRICS (Markdown Format)")
    print("="*60)
    markdown_table = format_as_markdown(results)
    print(markdown_table)
    
    # 同时输出到文件
    output_file = "combined_metrics.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# 模型评估指标\n\n")
        f.write(markdown_table)
        f.write("\n\n*注：表格中加粗行为多个数据集合并计算的结果*\n")
    
    print(f"\nMarkdown表格已保存到: {output_file}")


if __name__ == "__main__":
    main()