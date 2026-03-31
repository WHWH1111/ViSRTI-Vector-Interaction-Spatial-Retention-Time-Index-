#!/usr/bin/env python3
"""
将data1028_sheet0_pos_covered.csv转换为MMF_GNN_pos.csv格式的脚本
"""

import pandas as pd
import numpy as np
import os

dataset_type = 'neg' # pos or neg

def convert_csv(input_file, output_file):
    """转换CSV文件格式"""
    # 读取输入文件
    df = pd.read_csv(input_file)
    
    # 定义需要的列
    required_columns = [
        'Norman_SusDat_ID', 'Name', 'SMILES', 'Monoiso_Mass', 
        'logKow_EPISuite', 'Exp_logKow_EPISuite', 'alogp_ChemSpider', 
        'xlogp_ChemSpider', 'Koc_max_predicted (L/kg)',  # 使用Koc_max作为Koc_predicted
        'Tetrahymena_pyriformis_toxicity', 'Daphnia_toxicity', 
        'Algae_toxicity', 'Pimephales_promelas_toxicity', 
        'Prob. +ESI', 'Prob. -ESI'
    ]
    required_columns.append('Pred_RTI_Positive_ESI' if dataset_type == 'pos' else 'Pred_RTI_Negative_ESI')
    
    # 创建新的DataFrame，只包含需要的列
    new_df = df[required_columns].copy()
    
    # 重命名Koc_max_predicted (L/kg)列为Koc_predicted (L/kg)
    new_df.rename(columns={'Koc_max_predicted (L/kg)': 'Koc_predicted (L/kg)'}, inplace=True)
    
    # 保存到输出文件
    new_df.to_csv(output_file, index=False)
    print(f"转换完成，已保存到 {output_file}")
    print(f"共处理 {len(new_df)} 行数据")

def main():
    input_file = f"data/MMF-3/data1028_sheet0_{dataset_type}_covered.csv"
    output_file = f"data/MMF-3/MMF_GNN_{dataset_type}_test.csv"
    
    if not os.path.exists(input_file):
        print(f"错误：找不到输入文件 {input_file}")
        return
    
    convert_csv(input_file, output_file)

if __name__ == "__main__":
    main()