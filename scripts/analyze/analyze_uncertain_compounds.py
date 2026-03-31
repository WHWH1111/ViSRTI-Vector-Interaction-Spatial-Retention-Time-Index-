"""
1. neg
python scripts/analyze/analyze_uncertain_compounds.py --merge_files --files predictions\with-conf\mc_dropout_prediction_20251128-212827\test_MMF_GNN_neg_mc_dropout_predictions.csv predictions\with-conf\mc_dropout_prediction_20251128-225849\train_MMF_GNN_neg_mc_dropout_predictions.csv --rt-type neg

2. pos
python scripts/analyze/analyze_uncertain_compounds.py --merge_files --files predictions/with-conf/mc_dropout_prediction_20251127-040219/test_MMF_GNN_pos_mc_dropout_predictions.csv predictions/with-conf/mc_dropout_prediction_20251127-041109/train_MMF_GNN_pos_mc_dropout_predictions.csv --rt-type pos
"""

import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import argparse
import os
import glob
from datetime import datetime

def merge_prediction_files(file_pattern):
    """
    合并匹配指定模式的多个预测文件
    
    Parameters:
    file_pattern (str): 文件路径模式，如 "predictions/with-conf/*/test_MMF_GNN_neg_mc_dropout_predictions.csv"
    
    Returns:
    pd.DataFrame: 合并后的数据框
    """
    # 获取所有匹配的文件
    files = glob.glob(file_pattern)
    
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")
    
    print(f"Found {len(files)} files matching pattern:")
    for f in files:
        print(f"  - {f}")
    
    # 读取并合并所有文件
    dataframes = []
    for file in files:
        df = pd.read_csv(file)
        # 添加来源文件列
        df['source_file'] = os.path.basename(os.path.dirname(file))
        dataframes.append(df)
    
    # 合并所有数据框
    merged_df = pd.concat(dataframes, ignore_index=True)
    print(f"Merged {len(files)} files into a single dataframe with {len(merged_df)} rows")
    
    return merged_df

def merge_multiple_csv_files(file_list):
    """
    合并多个CSV文件
    
    Parameters:
    file_list (list): CSV文件路径列表
    
    Returns:
    pd.DataFrame: 合并后的数据框
    """
    if not file_list:
        raise ValueError("File list is empty")
    
    print(f"Merging {len(file_list)} CSV files:")
    for f in file_list:
        print(f"  - {f}")
    
    # 读取并合并所有文件
    dataframes = []
    for file in file_list:
        if not os.path.exists(file):
            print(f"Warning: File not found - {file}")
            continue
            
        df = pd.read_csv(file)
        # 添加来源文件列
        df['source_file'] = os.path.basename(file)
        dataframes.append(df)
    
    if not dataframes:
        raise FileNotFoundError("No valid files found to merge")
    
    # 合并所有数据框
    merged_df = pd.concat(dataframes, ignore_index=True)
    print(f"Merged {len(dataframes)} files into a single dataframe with {len(merged_df)} rows")
    
    return merged_df

def analyze_uncertain_compounds(predictions_file, tags_file, top_n=100, save_fig=True, merge_files=False, output_dir=None):
    """
    分析预测结果中具有高不确定性的化合物，并根据标签进行分类统计
    
    Parameters:
    predictions_file (str or list): 包含预测结果的CSV文件路径或文件模式或文件列表
    tags_file (str): 包含化合物标签的CSV文件路径
    top_n (int): 选取前N个高不确定性化合物的数量
    save_fig (bool): 是否保存图表
    merge_files (bool): 是否合并多个预测文件
    output_dir (str): 输出目录路径
    """
    
    # 处理文件合并逻辑
    if merge_files:
        if isinstance(predictions_file, str):
            # 处理文件模式匹配
            predictions_df = merge_prediction_files(predictions_file)
        elif isinstance(predictions_file, list):
            # 处理文件列表
            predictions_df = merge_multiple_csv_files(predictions_file)
        else:
            raise ValueError("predictions_file must be either a string pattern or a list of file paths")
    else:
        # 读取预测结果和标签数据
        predictions_df = pd.read_csv(predictions_file)
    
    tags_df = pd.read_csv(tags_file)
    
    # 创建SMILES到标签的映射字典
    smiles_to_tags = dict(zip(tags_df['SMILES'], tags_df['Tags']))
    
    # 添加标签列到预测数据中
    predictions_df['Tags'] = predictions_df['SMILES'].map(smiles_to_tags)
    
    # 删除没有标签的化合物
    predictions_with_tags = predictions_df.dropna(subset=['Tags'])
    
    # 根据Std值排序并选取top N (Std越低越好)
    top_std_df = predictions_with_tags.nsmallest(top_n, 'Std')
    
    # 根据Confidence值排序并选取top N (Confidence越高越好)
    top_conf_df = predictions_with_tags.nlargest(top_n, 'Confidence')
    
    # 分析基于Std的top N化合物的标签分布
    std_tags_counter = Counter()
    for tags_str in top_std_df['Tags']:
        tags_list = tags_str.split(',')
        std_tags_counter.update(tags_list)
    
    # 分析基于Confidence的top N化合物的标签分布
    conf_tags_counter = Counter()
    for tags_str in top_conf_df['Tags']:
        tags_list = tags_str.split(',')
        conf_tags_counter.update(tags_list)
    
    # 计算标签占比
    total_std_compounds = len(top_std_df)
    total_conf_compounds = len(top_conf_df)
    
    std_tag_percentages = {tag: count/total_std_compounds*100 
                          for tag, count in std_tags_counter.items()}
    conf_tag_percentages = {tag: count/total_conf_compounds*100 
                           for tag, count in conf_tags_counter.items()}
    
    # 按百分比降序排列
    sorted_std_tags = sorted(std_tag_percentages.items(), key=lambda x: x[1], reverse=True)
    sorted_conf_tags = sorted(conf_tag_percentages.items(), key=lambda x: x[1], reverse=True)
    
    # 打印结果
    print("Top {} compounds with highest Std values:".format(top_n))
    print("=" * 50)
    for i, (idx, compound_row) in enumerate(top_std_df.iterrows()):
        print(f"{i+1:3d}. SMILES: {compound_row['SMILES']}")
        print(f"      Std: {compound_row['Std']:.4f}, Confidence: {compound_row['Confidence']:.4f}")
        print(f"      Tags: {compound_row['Tags']}")
        if 'source_file' in compound_row:
            print(f"      Source: {compound_row['source_file']}")
        print()
    
    print("\n\nTag distribution for compounds with high Std:")
    print("=" * 50)
    for tag, percentage in sorted_std_tags:
        print(f"{tag}: {percentage:.2f}% ({std_tags_counter[tag]}/{total_std_compounds})")
        
    print("\n\nTop {} compounds with lowest Confidence values:".format(top_n))
    print("=" * 50)
    for i, (idx, compound_row) in enumerate(top_conf_df.iterrows()):
        print(f"{i+1:3d}. SMILES: {compound_row['SMILES']}")
        print(f"      Std: {compound_row['Std']:.4f}, Confidence: {compound_row['Confidence']:.4f}")
        print(f"      Tags: {compound_row['Tags']}")
        if 'source_file' in compound_row:
            print(f"      Source: {compound_row['source_file']}")
        print()
        
    print("\n\nTag distribution for compounds with low Confidence:")
    print("=" * 50)
    for tag, percentage in sorted_conf_tags:
        print(f"{tag}: {percentage:.2f}% ({conf_tags_counter[tag]}/{total_conf_compounds})")
    
    # 可视化标签分布
    if save_fig:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 绘制基于Std的标签分布
        tags_std, percentages_std = zip(*sorted_std_tags[:15])  # 只显示前15个标签
        bars1 = ax1.barh(range(len(tags_std)), percentages_std, color='skyblue')
        ax1.set_yticks(range(len(tags_std)))
        ax1.set_yticklabels(tags_std)
        ax1.set_xlabel('Percentage (%)')
        ax1.set_title('Tag Distribution (Top {} compounds with highest Std)'.format(top_n))
        ax1.invert_yaxis()  # 最高百分比在上方
        
        # 添加数值标签
        for i, (bar, v) in enumerate(zip(bars1, percentages_std)):
            ax1.text(v + 0.5, i, f"{v:.1f}%", va='center')
        
        # 绘制基于Confidence的标签分布
        tags_conf, percentages_conf = zip(*sorted_conf_tags[:15])  # 只显示前15个标签
        bars2 = ax2.barh(range(len(tags_conf)), percentages_conf, color='lightcoral')
        ax2.set_yticks(range(len(tags_conf)))
        ax2.set_yticklabels(tags_conf)
        ax2.set_xlabel('Percentage (%)')
        ax2.set_title('Tag Distribution (Top {} compounds with lowest Confidence)'.format(top_n))
        ax2.invert_yaxis()  # 最高百分比在上方
        
        # 添加数值标签
        for i, (bar, v) in enumerate(zip(bars2, percentages_conf)):
            ax2.text(v + 0.5, i, f"{v:.1f}%", va='center')
        
        plt.tight_layout()
        
        # 确定图像保存路径
        if output_dir:
            figure_path = os.path.join(output_dir, 'uncertain_compounds_tag_distribution.png')
        else:
            figure_path = 'uncertain_compounds_tag_distribution.png'
            
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        print(f"\nTag distribution chart saved to '{figure_path}'")
    
    return top_std_df, top_conf_df, sorted_std_tags, sorted_conf_tags


def main():
    parser = argparse.ArgumentParser(description='Analyze uncertain compounds based on Std and Confidence values')
    parser.add_argument('--top_n', type=int, default=100, help='Number of top compounds to analyze')
    parser.add_argument('--no_fig', action='store_true', help='Do not save figures')
    parser.add_argument('--merge_files', action='store_true', help='Merge multiple prediction files')
    parser.add_argument('--pattern', type=str, 
                        default="predictions/with-conf/*/test_MMF_GNN_neg_mc_dropout_predictions.csv",
                        help='File pattern for merging multiple prediction files')
    parser.add_argument('--files', nargs='+', 
                        help='List of specific CSV files to merge')
    parser.add_argument('--rt-type', choices=['pos', 'neg'], default='neg',
                        help='RT type (pos or neg)')
    
    args = parser.parse_args()
    
    # 确定要处理的预测文件
    if args.files:
        # 如果提供了具体的文件列表，则使用这些文件
        predictions_file = args.files
        merge_files = True
    elif args.merge_files:
        # 如果指定了合并文件但没有提供具体文件列表，则使用模式匹配
        # 根据rt-type替换pattern中的pos/neg
        predictions_file = args.pattern.replace('neg', args.rt_type)
        merge_files = True
    else:
        # 默认使用单个文件
        default_file = f"predictions/with-conf/mc_dropout_prediction_20251127-031253/test_MMF_GNN_{args.rt_type}_mc_dropout_predictions.csv"
        predictions_file = default_file
        merge_files = False
        
    tags_file = f"data/MMF-3/tags/{args.rt_type}_compound_tags.csv"
    
    
    # 检查文件是否存在
    if not merge_files:
        try:
            open(predictions_file)
            open(tags_file)
        except FileNotFoundError as e:
            print(f"Error: File not found - {e}")
            return
    
    # 创建输出目录结构
    base_output_dir = os.path.join("results", "uncertain")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    final_output_dir = os.path.join(base_output_dir, timestamp)
    os.makedirs(final_output_dir, exist_ok=True)
    
    # 执行分析
    top_std_df, top_conf_df, std_tags, conf_tags = analyze_uncertain_compounds(
        predictions_file, 
        tags_file, 
        top_n=args.top_n, 
        save_fig=not args.no_fig,
        merge_files=merge_files,
        output_dir=final_output_dir
    )
    
    # 保存结果到文件
    std_csv_path = os.path.join(final_output_dir, 'top100_by_std_with_tags.csv')
    conf_csv_path = os.path.join(final_output_dir, 'top100_by_conf_with_tags.csv')
    
    top_std_df.to_csv(std_csv_path, index=False)
    top_conf_df.to_csv(conf_csv_path, index=False)
    
    print(f"\nDetailed results saved to '{std_csv_path}' and '{conf_csv_path}'")

if __name__ == "__main__":
    main()