import pandas as pd
import os
from pathlib import Path

def convert_csv_to_excel(input_dir, output_dir=None):
    """
    将指定目录下的所有CSV文件转换为Excel文件
    
    Parameters:
    input_dir (str): 包含CSV文件的目录路径
    output_dir (str): 输出Excel文件的目录路径，默认为None，表示与输入目录相同
    """
    # 获取输入目录的Path对象
    input_path = Path(input_dir)
    
    # 如果没有指定输出目录，则使用输入目录
    if output_dir is None:
        output_path = input_path
    else:
        output_path = Path(output_dir)
        # 确保输出目录存在
        output_path.mkdir(parents=True, exist_ok=True)
    
    # 查找所有CSV文件
    csv_files = list(input_path.glob("*.csv"))
    
    if not csv_files:
        print(f"在 {input_dir} 目录中未找到CSV文件")
        return
    
    print(f"找到 {len(csv_files)} 个CSV文件:")
    
    for csv_file in csv_files:
        try:
            # 读取CSV文件
            print(f"正在处理: {csv_file.name}")
            df = pd.read_csv(csv_file)
            
            # 生成输出文件名
            excel_filename = csv_file.stem + ".xlsx"
            excel_filepath = output_path / excel_filename
            
            # 保存为Excel文件
            df.to_excel(excel_filepath, index=False)
            print(f"已保存: {excel_filepath}")
            
        except Exception as e:
            print(f"处理 {csv_file.name} 时出错: {str(e)}")
    
    print("转换完成!")

if __name__ == "__main__":
    # 设置输入目录
    input_directory = "data/MMF-3/tags"
    
    # 执行转换
    convert_csv_to_excel(input_directory)