import pandas as pd
import os
import sys
import zipfile

def check_xlsx_file(file_path):
    """
    检查Excel文件的完整性
    """
    try:
        # 检查是否为有效的zip文件（.xlsx文件本质上是zip压缩包）
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            print(f"Excel文件包含以下文件: {file_list[:10]}")  # 只显示前10个文件
            if len(file_list) > 10:
                print(f"... 还有 {len(file_list) - 10} 个文件")
            return True
    except zipfile.BadZipFile:
        print("错误: 文件不是有效的ZIP文件格式")
        return False
    except Exception as e:
        print(f"检查文件时出错: {e}")
        return False

def convert_xlsx_to_csv(xlsx_file_path, csv_file_path=None):
    """
    将Excel文件(.xlsx)转换为CSV文件
    
    参数:
    xlsx_file_path (str): Excel文件路径
    csv_file_path (str, optional): CSV文件输出路径，默认为同名文件但扩展名为.csv
    """
    # 如果未指定输出文件路径，则自动生成
    if csv_file_path is None:
        base_name = os.path.splitext(xlsx_file_path)[0]
        csv_file_path = base_name + '.csv'
    
    try:
        # 检查文件是否存在
        if not os.path.exists(xlsx_file_path):
            print(f"错误: 文件 {xlsx_file_path} 不存在")
            return
        
        # 获取文件大小
        file_size = os.path.getsize(xlsx_file_path)
        print(f"正在处理文件: {xlsx_file_path} (大小: {file_size} 字节)")
        
        # 检查文件扩展名
        _, ext = os.path.splitext(xlsx_file_path)
        if ext.lower() not in ['.xlsx', '.xls']:
            print(f"警告: 文件扩展名 {ext} 可能不是有效的Excel格式")
        
        # 检查文件完整性
        print("检查Excel文件完整性...")
        if not check_xlsx_file(xlsx_file_path):
            print("文件完整性检查失败，请确认文件未损坏")
            return
        
        # 尝试多种方式读取Excel文件
        df = None
        engines = ['openpyxl', 'xlrd']
        if ext.lower() == '.xls':
            engines = ['xlrd', 'openpyxl']
        
        last_error = None
        for engine in engines:
            try:
                print(f"尝试使用 {engine} 引擎读取...")
                if engine == 'xlrd':
                    # xlrd对.xlsx支持有限，主要用于.xls
                    if ext.lower() == '.xlsx':
                        print("跳过xlrd引擎，因为它不支持.xlsx文件")
                        continue
                    df = pd.read_excel(xlsx_file_path, engine=engine)
                else:
                    df = pd.read_excel(xlsx_file_path, engine=engine)
                print(f"使用 {engine} 引擎成功读取文件")
                break
            except Exception as e:
                print(f"使用 {engine} 引擎失败: {e}")
                last_error = e
        
        if df is None:
            raise Exception(f"无法使用任何引擎读取文件: {last_error}")
        
        # 显示一些基本信息
        print(f"数据形状: {df.shape}")
        if not df.empty:
            print(f"列名: {list(df.columns)}")
            print("前几行数据:")
            print(df.head())
        else:
            print("警告: 数据文件为空")
        
        # 保存为CSV文件
        df.to_csv(csv_file_path, index=False, encoding='utf-8')
        print(f"成功将 {xlsx_file_path} 转换为 {csv_file_path}")
        
    except Exception as e:
        print(f"转换失败: {e}")
        # 提供更多调试信息
        import traceback
        print("详细错误信息:")
        traceback.print_exc()

def convert_all_xlsx_in_directory(directory_path):
    """
    将目录中的所有xlsx文件转换为csv文件
    
    参数:
    directory_path (str): 包含xlsx文件的目录路径
    """
    try:
        files = os.listdir(directory_path)
        xlsx_files = [f for f in files if f.endswith('.xlsx') or f.endswith('.xls')]
        
        if not xlsx_files:
            print(f"目录 {directory_path} 中未找到Excel文件")
            return
            
        for xlsx_file in xlsx_files:
            xlsx_path = os.path.join(directory_path, xlsx_file)
            csv_path = os.path.join(directory_path, os.path.splitext(xlsx_file)[0] + '.csv')
            print(f"\n处理文件: {xlsx_file}")
            convert_xlsx_to_csv(xlsx_path, csv_path)
            
    except Exception as e:
        print(f"处理目录时出错: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法:")
        print("  python convert_xlsx_to_csv.py <xlsx文件路径> [csv文件路径]")
        print("  python convert_xlsx_to_csv.py <目录路径> --all")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    if len(sys.argv) >= 3 and sys.argv[2] == "--all":
        # 转换目录中的所有xlsx文件
        convert_all_xlsx_in_directory(input_path)
    elif len(sys.argv) == 3:
        # 转换单个文件并指定输出路径
        convert_xlsx_to_csv(input_path, sys.argv[2])
    else:
        # 转换单个文件使用默认输出路径
        convert_xlsx_to_csv(input_path)