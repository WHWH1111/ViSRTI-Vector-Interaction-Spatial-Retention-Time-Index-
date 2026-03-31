"""
This script compares the predictions made by a model with the original data. It checks whether the SMILES strings and actual values match between the two datasets. It also calculates the difference between the predicted and original values and provides statistics about the differences.

Usage:
    python compare_predictions_with_original.py

Note:
    - The script assumes that the prediction file and the original data file are in the same directory.
    - The prediction file should contain the predicted values and the SMILES strings.
"""

import pandas as pd
import os
import numpy as np

prediction_name = 'prediction_20251122-115654'

def compare_predictions_with_original():
    # 定义文件路径
    prediction_file = rf"predictions\{prediction_name}\None_data1028_sheet0_predictions.csv"
    original_file = r"data\MMF-3\data1028_sheet0.csv"
    
    # 检查文件是否存在
    if not os.path.exists(prediction_file):
        print(f"预测文件不存在: {prediction_file}")
        return
    
    if not os.path.exists(original_file):
        print(f"原始数据文件不存在: {original_file}")
        return
    
    # 读取预测文件
    pred_df = pd.read_csv(prediction_file)
    print(f"预测文件包含 {len(pred_df)} 行数据")
    print("预测文件列名:", pred_df.columns.tolist())
    
    # 读取原始数据文件
    orig_df = pd.read_csv(original_file)
    print(f"原始数据文件包含 {len(orig_df)} 行数据")
    print("原始数据文件列名:", orig_df.columns.tolist())
    
    # 检查必要的列是否存在
    required_pred_columns = ['Norman_SusDat_ID', 'SMILES', 'Actual']
    missing_pred_columns = [col for col in required_pred_columns if col not in pred_df.columns]
    if missing_pred_columns:
        print(f"预测文件缺少必要的列: {missing_pred_columns}")
        return
    
    # 检查原始文件中必要的列
    # required_orig_columns = ['Norman_SusDat_ID', 'SMILES', 'Pred_RTI_Positive_ESI'] # INFO pos
    required_orig_columns = ['Norman_SusDat_ID', 'SMILES', 'Pred_RTI_Negative_ESI'] # INFO neg
    missing_orig_columns = [col for col in required_orig_columns if col not in orig_df.columns]
    if missing_orig_columns:
        print(f"原始文件缺少必要的列: {missing_orig_columns}")
        return
    
    # 设置ID为索引以便匹配
    pred_df.set_index('Norman_SusDat_ID', inplace=True)
    orig_df.set_index('Norman_SusDat_ID', inplace=True)
    
    # 统计匹配情况
    matched_ids = pred_df.index.intersection(orig_df.index)
    print(f"\n匹配的ID数量: {len(matched_ids)}")
    print(f"预测文件中未在原始文件中找到的ID数量: {len(pred_df.index.difference(orig_df.index))}")
    print(f"原始文件中未在预测文件中找到的ID数量: {len(orig_df.index.difference(pred_df.index))}")
    
    # 创建一个结果DataFrame来存储比较结果
    comparison_results = []
    
    # 检查SMILES和Actual值是否对应到原始文件的Pred_RTI_Positive_ESI字段
    for id_val in matched_ids:
        pred_smiles = pred_df.loc[id_val, 'SMILES']
        pred_actual = pred_df.loc[id_val, 'Actual']
        orig_smiles = orig_df.loc[id_val, 'SMILES']
        # orig_rti = orig_df.loc[id_val, 'Pred_RTI_Positive_ESI']
        orig_rti = orig_df.loc[id_val, 'Pred_RTI_Negative_ESI']
        
        # 检查SMILES是否匹配
        smiles_match = (pred_smiles == orig_smiles)
        
        # 检查Actual值是否匹配（考虑精度差异）
        # 原始数据只保留两位小数，所以我们将预测值也四舍五入到两位小数进行比较
        actual_match = (round(pred_actual, 2) == orig_rti)
        
        # 计算差异
        difference = abs(pred_actual - orig_rti)
        
        comparison_results.append({
            'ID': id_val,
            'Pred_SMILES': pred_smiles,
            'Orig_SMILES': orig_smiles,
            'SMILES_Match': smiles_match,
            'Pred_Actual': pred_actual,
            'Orig_RTI': orig_rti,
            'Rounded_Pred_Actual': round(pred_actual, 2),
            'Actual_Match': actual_match,
            'Difference': difference
        })
    
    # 转换为DataFrame
    comparison_df = pd.DataFrame(comparison_results)
    
    # 统计匹配情况
    smiles_matches = comparison_df['SMILES_Match'].sum()
    actual_matches = comparison_df['Actual_Match'].sum()
    
    print(f"\n详细比较结果:")
    print(f"匹配的SMILES数量: {smiles_matches}/{len(matched_ids)} ({smiles_matches/len(matched_ids)*100:.2f}%)")
    print(f"考虑精度差异后匹配的Actual值数量: {actual_matches}/{len(matched_ids)} ({actual_matches/len(matched_ids)*100:.2f}%)")
    
    # 显示一些不匹配的例子
    mismatched_smiles = comparison_df[~comparison_df['SMILES_Match']]
    if len(mismatched_smiles) > 0:
        print(f"\n发现 {len(mismatched_smiles)} 个SMILES不匹配的例子:")
        print(mismatched_smiles[['ID', 'Pred_SMILES', 'Orig_SMILES']].head(10))
    
    mismatched_actual = comparison_df[~comparison_df['Actual_Match']]
    if len(mismatched_actual) > 0:
        print(f"\n发现 {len(mismatched_actual)} 个Actual值与原始RTI值不匹配的例子 (考虑精度差异后):")
        print(mismatched_actual[['ID', 'Pred_Actual', 'Rounded_Pred_Actual', 'Orig_RTI', 'Difference']].head(10))
    else:
        print("\n所有Actual值在考虑精度差异后都与原始RTI值匹配!")
    
    # 显示差异统计
    print(f"\n差异统计:")
    print(f"平均差异: {comparison_df['Difference'].mean():.6f}")
    print(f"最大差异: {comparison_df['Difference'].max():.6f}")
    print(f"最小差异: {comparison_df['Difference'].min():.6f}")
    
    # 查看差异大于0.01的例子
    large_diff = comparison_df[comparison_df['Difference'] > 0.01]
    if len(large_diff) > 0:
        print(f"\n发现 {len(large_diff)} 个差异大于0.01的例子:")
        print(large_diff[['ID', 'Pred_Actual', 'Rounded_Pred_Actual', 'Orig_RTI', 'Difference']].head(10))
    else:
        print("\n所有差异都小于等于0.01!")
    
    # 保存详细结果到CSV文件
    # output_file = "predictions_vs_original_comparison.csv"
    # comparison_df.to_csv(output_file, index=False)
    # print(f"\n详细比较结果已保存到: {output_file}")

if __name__ == "__main__":
    compare_predictions_with_original()