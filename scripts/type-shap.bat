#!/usr/bin/env pwsh

@echo off
setlocal enabledelayedexpansion

:: 化合物类型定义
set compoundTypes=Aromatic Aliphatic Nitrogen_containing Oxygen_containing Sulfur_containing Phosphorus_containing Halogen_containing High_hydrophobic Medium_hydrophobic Hydrophilic

:: 输出提示信息
echo Available compound types:
for %%t in (%compoundTypes%) do (
    echo - %%t
)

:: 遍历所有化合物类型并执行SHAP分析
for %%t in (%compoundTypes%) do (
    echo.
    echo ========================================
    echo Processing compound type: %%t
    echo ========================================
    
    python analyze_shap.py ^
      --model-path log/data-2/pos-3-train_20251104-055350_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/model.pt ^
      --training-params-path log/data-2/pos-3-train_20251104-055350_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/training_params.json ^
      --data-path ./data/MMF-3/ ^
      --dataset-name MMF_GNN_pos ^
      --sample-size 10 ^
      --background-samples 5 ^
      --batch-size 16 ^
      --output-dir ./results/shap/pos-3-type-shap/%%t ^
      --use-train-data ^
      --filter-by-tag %%t
    
    echo Completed processing for compound type: %%t
)

echo.
echo ========================================
echo All compound types processed successfully!
echo ========================================