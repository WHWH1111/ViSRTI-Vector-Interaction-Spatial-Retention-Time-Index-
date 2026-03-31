import os
import pandas as pd
import argparse
from pathlib import Path


def merge_predictions_with_tags(predictions_dir, tags_file, output_file):
    """
    合并指定目录下所有子文件夹中的预测CSV文件，根据ID排序，
    并根据SMILES将标签信息补充到合并后的CSV文件中
    
    Args:
        predictions_dir (str): 包含预测结果子文件夹的目录路径
        tags_file (str): 包含SMILES和标签信息的CSV文件路径
        output_file (str): 输出文件路径
    """
    # 收集所有预测CSV文件
    csv_files = []
    for root, dirs, files in os.walk(predictions_dir):
        for file in files:
            if file.endswith('_predictions.csv'):
                csv_files.append(os.path.join(root, file))
    
    if not csv_files:
        print(f"在 {predictions_dir} 中没有找到预测CSV文件")
        return
    
    # 读取并合并所有预测文件
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dfs.append(df)
    
    # 合并所有数据框
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # 根据ID排序 (Norman_SusDat_ID)
    merged_df = merged_df.sort_values('Norman_SusDat_ID').reset_index(drop=True)
    
    # 读取标签文件
    tags_df = pd.read_csv(tags_file)
    
    # 根据SMILES合并标签信息
    final_df = pd.merge(merged_df, tags_df, on='SMILES', how='left')
    
    # 保存结果
    final_df.to_csv(output_file, index=False)
    print(f"合并完成，共处理 {len(final_df)} 条记录")
    print(f"结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='合并预测结果并添加标签信息')
    parser.add_argument('--predictions_dir', type=str, 
                        default='predictions/visnet-v2-5fold/pos',
                        help='包含预测结果子文件夹的目录路径')
    parser.add_argument('--tags_file', type=str,
                        default='data/MMF-3/tags/pos_compound_tags.csv',
                        help='包含SMILES和标签信息的CSV文件路径')
    parser.add_argument('--output_file', type=str,
                        default='merged_predictions_with_tags.csv',
                        help='输出文件路径')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    merge_predictions_with_tags(args.predictions_dir, args.tags_file, args.output_file)


if __name__ == "__main__":
    main()