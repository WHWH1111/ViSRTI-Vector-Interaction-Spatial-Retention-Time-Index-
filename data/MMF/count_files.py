#!/usr/bin/env python3
"""
统计MMF目录下所有CSV文件的详细信息
"""
import os
import pandas as pd
from pathlib import Path

def count_csv_files():
    """统计当前目录下所有CSV文件的信息"""
    # 获取当前目录
    current_dir = Path(__file__).parent
    print(f"统计目录: {current_dir}")
    print("=" * 80)
    
    # 获取所有CSV文件
    csv_files = list(current_dir.glob("*.csv"))
    
    if not csv_files:
        print("当前目录下没有找到CSV文件")
        return
    
    # 存储统计信息
    total_lines = 0
    total_size = 0
    
    # 按文件名排序
    csv_files.sort()
    
    print(f"{'文件名':<40} {'行数':<10} {'大小(MB)':<12} {'列数':<8}")
    print("-" * 80)
    
    for csv_file in csv_files:
        try:
            # 获取文件大小
            file_size = csv_file.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            total_size += file_size
            
            # 读取文件获取行数和列数
            df = pd.read_csv(csv_file)
            rows, cols = df.shape
            
            # CSV文件包含标题行，所以行数需要+1
            total_lines += rows + 1
            
            print(f"{csv_file.name:<40} {rows+1:<10} {file_size_mb:<12.2f} {cols:<8}")
        except Exception as e:
            print(f"{csv_file.name:<40} 错误: {str(e)}")
    
    print("-" * 80)
    print(f"{'总计':<40} {total_lines:<10} {total_size/(1024*1024):<12.2f} {'':<8}")
    print(f"文件数量: {len(csv_files)}")

if __name__ == "__main__":
    count_csv_files()