#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按化合物类型执行SHAP分析的脚本

CUDA_VISIBLE_DEVICES=2 python scripts/type-shap.py
"""

import subprocess
import sys
import os

# SHAP分析参数配置
# 1. neg
model_dir = "visnet-v2/neg-3-mask(koc)-train_20251128-162233_dim48_layerH6_layerO6_batch64_lr0.0001_iter150"
MODEL_PATH = f"log/{model_dir}/model.pt"
TRAINING_PARAMS_PATH = f"log/{model_dir}/training_params.json"
DATA_PATH = "./data/MMF-3/"
DATASET_NAME = "MMF_GNN_neg"
SAMPLE_SIZE = "200"
BACKGROUND_SAMPLES = "50"
BATCH_SIZE = "256"
output_dir = "neg-3-type-shap"

# 2. pos
# model_dir = "visnet-v2/pos-3-mask(koc)-train_20251126-172959_dim48_layerH6_layerO6_batch64_lr0.0001_iter150"
# MODEL_PATH = f"log/{model_dir}/model.pt"
# TRAINING_PARAMS_PATH = f"log/{model_dir}/training_params.json"
# DATA_PATH = "./data/MMF-3/"
# DATASET_NAME = "MMF_GNN_pos"
# SAMPLE_SIZE = "200"
# BACKGROUND_SAMPLES = "50"
# BATCH_SIZE = "2048"
# output_dir = "pos-3-type-shap"

# 化合物类型定义
COMPOUND_TYPES = [
    'Aromatic',           # 芳香化合物
    'Aliphatic',          # 脂肪族化合物  
    'Nitrogen_containing', # 含氮化合物（包括胺类、酰胺类、硝基化合物等）
    'Oxygen_containing',   # 含氧化合物（包括醇/酚类、羧酸类、羰基化合物等）
    'Sulfur_containing',  # 含硫化合物
    'Phosphorus_containing', # 含磷化合物
    'Halogen_containing', # 含卤化合物
    'High_hydrophobic',   # 高疏水性（logKow > 4）
    'Medium_hydrophobic', # 中等疏水性（2 < logKow <= 4）
    'Hydrophilic'         # 亲水性（logKow <= 2）
]

def run_shap_analysis(compound_type):
    """
    对指定化合物类型运行SHAP分析
    """
    print(f"\n{'='*40}")
    print(f"Processing compound type: {compound_type}")
    print(f"{'='*40}")
    
    # 构建命令
    cmd = [
        "python", "analyze_shap.py",
        "--model-path", MODEL_PATH,
        "--training-params-path", TRAINING_PARAMS_PATH,
        "--data-path", DATA_PATH,
        "--dataset-name", DATASET_NAME,
        "--sample-size", SAMPLE_SIZE,
        "--background-samples", BACKGROUND_SAMPLES,
        "--batch-size", BATCH_SIZE,
        "--output-dir", f"./results/shap/{output_dir}/{compound_type}",
        "--use-train-data",
        "--filter-by-tag", compound_type
    ]
    
    print("Executing command:")
    print(" ".join(cmd))
    
    # 执行命令并实时显示输出
    try:
        # 使用Popen以便实时显示输出，直接连接到主进程的stdout/stderr以确保进度条正确显示
        process = subprocess.Popen(cmd)
        
        # 等待进程完成
        return_code = process.wait()
        
        if return_code == 0:
            print(f"Completed processing for compound type: {compound_type}")
            return True
        else:
            print(f"Error processing compound type {compound_type}. Return code: {return_code}")
            return False
            
    except Exception as e:
        print(f"Error processing compound type {compound_type}: {e}")
        return False

def main():
    print("Available compound types:")
    for compound_type in COMPOUND_TYPES:
        print(f"- {compound_type}")
    
    # 遍历所有化合物类型并执行SHAP分析
    failed_types = []
    for compound_type in COMPOUND_TYPES:
        success = run_shap_analysis(compound_type)
        if not success:
            failed_types.append(compound_type)
    
    print(f"\n{'='*40}")
    if failed_types:
        print(f"Some compound types failed: {', '.join(failed_types)}")
    else:
        print("All compound types processed successfully!")
    print(f"{'='*40}")

if __name__ == "__main__":
    main()