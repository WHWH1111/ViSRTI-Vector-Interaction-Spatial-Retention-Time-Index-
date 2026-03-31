import pandas as pd
import os
import argparse

def merge_complete_data(input_pred, input_neg, input_pos, output_file):
    """
    合并预测数据和原始数据文件中的完整信息
    
    参数:
    input_pred : str
        合并后的预测数据文件路径
    input_neg : str
        负离子原始数据文件路径
    input_pos : str
        正离子原始数据文件路径
    output_file : str
        输出文件路径
    """
    # 读取文件
    pred_df = pd.read_csv(input_pred)
    neg_df = pd.read_csv(input_neg)
    pos_df = pd.read_csv(input_pos)
    
    # 重命名列以区分正负离子数据
    neg_columns = {col: f"{col}_neg" for col in neg_df.columns if col != 'Norman_SusDat_ID'}
    pos_columns = {col: f"{col}_pos" for col in pos_df.columns if col != 'Norman_SusDat_ID'}
    
    neg_df.rename(columns=neg_columns, inplace=True)
    pos_df.rename(columns=pos_columns, inplace=True)
    
    # 合并数据
    merged_df = pred_df.merge(neg_df, left_on='Norman_SusDat_ID', right_on='Norman_SusDat_ID', how='left')
    merged_df = merged_df.merge(pos_df, left_on='Norman_SusDat_ID', right_on='Norman_SusDat_ID', how='left')
    
    # 保存结果
    merged_df.to_csv(output_file, index=False)
    print(f"Complete dataset saved to {output_file}")
    
    # 显示一些统计信息
    total_compounds = len(merged_df)
    compounds_with_neg_data = merged_df['SMILES_neg'].notna().sum()
    compounds_with_pos_data = merged_df['SMILES_pos'].notna().sum()
    
    print(f"Total compounds: {total_compounds}")
    print(f"Compounds with negative ion data: {compounds_with_neg_data}")
    print(f"Compounds with positive ion data: {compounds_with_pos_data}")

def main():
    parser = argparse.ArgumentParser(description='Merge complete data from prediction and original datasets')
    parser.add_argument('--input-pred', default='predictions/visnet-v2-5fold/merged_neg_pos_predictions_by_id.csv',
                        help='Input prediction file path')
    parser.add_argument('--input-neg', default='data/MMF-3/MMF_GNN_neg.csv',
                        help='Input negative ion data file path')
    parser.add_argument('--input-pos', default='data/MMF-3/MMF_GNN_pos.csv',
                        help='Input positive ion data file path')
    parser.add_argument('--output', default='data/MMF-3/complete_dataset.csv',
                        help='Output file path')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    merge_complete_data(args.input_pred, args.input_neg, args.input_pos, args.output)

if __name__ == "__main__":
    main()