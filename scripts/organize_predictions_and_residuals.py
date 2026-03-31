import pandas as pd
import argparse
from pathlib import Path
import os


def organize_predictions_and_residuals(input_file, output_file):
    """
    整理预测值和残差数据，按指定格式输出到CSV文件
    
    Args:
        input_file (str): 输入的预测结果CSV文件路径
        output_file (str): 输出整理后的CSV文件路径
    """
    # 读取输入文件
    df = pd.read_csv(input_file)
    
    # 计算残差 (Actual - Predicted)
    df['Residual'] = df['Actual'] - df['Predicted']
    
    # 获取所有唯一标签类别
    all_tags = set()
    for tags_str in df['Tags']:
        tags = tags_str.split(',')
        all_tags.update(tags)
    
    all_tags = sorted(all_tags)
    
    # 创建输出DataFrame的列
    output_columns = ['pos全部物质_预测值', 'pos全部物质_残差']
    for tag in all_tags:
        output_columns.extend([f'{tag}_预测值', f'{tag}_残差'])
    
    # 为每种标签类别创建独立的数据列表
    output_data = {}
    
    # 添加全部物质的数据
    output_data['pos全部物质_预测值'] = df['Predicted'].tolist()
    output_data['pos全部物质_残差'] = df['Residual'].tolist()
    
    # 为每种标签类别分别创建预测值和残差列
    for tag in all_tags:
        # 筛选属于该标签的化合物
        tag_mask = df['Tags'].str.contains(tag)
        tag_data = df[tag_mask]
        
        # 添加该标签的预测值和残差
        output_data[f'{tag}_预测值'] = tag_data['Predicted'].tolist()
        output_data[f'{tag}_残差'] = tag_data['Residual'].tolist()
    
    # 创建输出DataFrame，每列独立排列
    output_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in output_data.items()]))
    
    # 保存到CSV文件
    output_df.to_csv(output_file, index=False)
    print(f"整理完成，结果已保存到: {output_file}")
    
    # 显示一些统计信息
    print("\n标签类别:")
    for i, tag in enumerate(all_tags, 1):
        count = len(df[df['Tags'].str.contains(tag)])
        print(f"{i}. {tag}: {count}个化合物")


def main():
    parser = argparse.ArgumentParser(description='整理预测值和残差数据')
    parser.add_argument('--input_file', type=str,
                        default='predictions/visnet-v2-5fold/pos/merged_predictions_with_tags.csv',
                        help='输入的预测结果CSV文件路径')
    parser.add_argument('--output_file', type=str,
                        help='输出整理后的CSV文件路径')
    
    args = parser.parse_args()
    
    # 如果未指定输出文件，则将其保存到与输入文件相同的目录中
    if args.output_file is None:
        input_path = Path(args.input_file)
        args.output_file = str(input_path.parent / "organized_predictions_residuals.csv")
    
    # 确保输出目录存在
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    organize_predictions_and_residuals(args.input_file, args.output_file)


if __name__ == "__main__":
    main()