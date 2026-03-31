import pandas as pd
import os
import argparse
from pathlib import Path


def csv_to_xlsx(csv_file_path, xlsx_file_path=None):
    """
    将 CSV 文件转换为 Excel (XLSX) 文件
    
    Parameters:
    csv_file_path (str): 输入的 CSV 文件路径
    xlsx_file_path (str): 输出的 XLSX 文件路径，默认为 None，表示使用相同文件名但扩展名为 .xlsx
    """
    # 如果没有指定输出文件路径，则自动生成
    if xlsx_file_path is None:
        xlsx_file_path = Path(csv_file_path).with_suffix('.xlsx')
    
    # 读取 CSV 文件
    df = pd.read_csv(csv_file_path)
    
    # 写入 Excel 文件
    df.to_excel(xlsx_file_path, index=False)
    print(f"已将 {csv_file_path} 转换为 {xlsx_file_path}")


def process_directory(directory_path):
    """
    处理指定目录下的所有 CSV 文件，将其转换为 XLSX 格式
    
    Parameters:
    directory_path (str): 包含 CSV 文件的目录路径
    """
    directory = Path(directory_path)
    
    # 检查目录是否存在
    if not directory.exists():
        print(f"错误: 目录 {directory_path} 不存在")
        return
    
    # 查找所有 CSV 文件
    csv_files = list(directory.glob("*.csv"))
    
    if not csv_files:
        print(f"在目录 {directory_path} 中未找到 CSV 文件")
        return
    
    print(f"在目录 {directory_path} 中找到 {len(csv_files)} 个 CSV 文件")
    
    # 转换每个 CSV 文件
    for csv_file in csv_files:
        xlsx_file = csv_file.with_suffix('.xlsx')
        try:
            csv_to_xlsx(str(csv_file), str(xlsx_file))
        except Exception as e:
            print(f"转换文件 {csv_file} 时出错: {e}")


def main():
    parser = argparse.ArgumentParser(description='将 CSV 文件转换为 Excel (XLSX) 格式')
    parser.add_argument('path', nargs='?', default='.', 
                        help='要处理的文件或目录路径（默认为当前目录）')
    parser.add_argument('--file', '-f', dest='file_path',
                        help='指定单个 CSV 文件进行转换')
    
    args = parser.parse_args()
    
    # 如果指定了单个文件
    if args.file_path:
        if os.path.isfile(args.file_path) and args.file_path.endswith('.csv'):
            csv_to_xlsx(args.file_path)
        else:
            print(f"错误: {args.file_path} 不是一个有效的 CSV 文件")
    # 如果提供了路径且是目录
    elif os.path.isdir(args.path):
        process_directory(args.path)
    # 如果提供了路径且是文件
    elif os.path.isfile(args.path) and args.path.endswith('.csv'):
        csv_to_xlsx(args.path)
    else:
        print(f"错误: {args.path} 不是一个有效的目录或 CSV 文件")


if __name__ == "__main__":
    main()