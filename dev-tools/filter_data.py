import pandas as pd

def filter_data_separate_and_combined(input_file):
    """
    从MMF-GNN.csv文件中进行七种过滤：
    1. 仅Uncertainty_RTI_pos为"Covered by Model"的条目
    2. 仅Uncertainty_RTI_pos不为"Covered by Model"的条目
    3. 仅Uncertainty_RTI_neg为"Covered by Model"的条目
    4. 仅Uncertainty_RTI_neg不为"Covered by Model"的条目
    5. 同时满足两个条件的条目
    6. 都不满足条件的条目
    7. 任意一个字段不为"Covered by Model"或"Covered by chemical space of the model"的条目
    
    Parameters:
    input_file (str): 输入CSV文件路径
    
    Returns:
    tuple: (rti_pos_covered_df, rti_pos_not_covered_df, 
            rti_neg_covered_df, rti_neg_not_covered_df, 
            both_covered_df, neither_df, uncovered_df)
    """
    # 读取CSV文件
    df = pd.read_csv(input_file)
    
    # 显示数据基本信息
    print(f"原始数据集大小: {df.shape}")
    
    # 定义有效值列表
    valid_values = ['Covered by Model', 'Covered by chemical space of the model']
    
    # 筛选出任意一个字段不为有效值的条目
    uncovered_df = df[~((df['Uncertainty_RTI_pos'].isin(valid_values)) & 
                        (df['Uncertainty_RTI_neg'].isin(valid_values)))]
    
    # 1. 仅Uncertainty_RTI_pos为"Covered by Model"的条目
    rti_pos_covered_df = df[df['Uncertainty_RTI_pos'] == 'Covered by Model']

    # 1.2 Uncertainty_RTI_pos不为"Covered by Model"的条目
    rti_pos_not_covered_df = df[df['Uncertainty_RTI_pos'] != 'Covered by Model']

    # 2. 仅Uncertainty_RTI_neg为"Covered by Model"的条目
    rti_neg_covered_df = df[df['Uncertainty_RTI_neg'] == 'Covered by Model']

    # 2.2 Uncertainty_RTI_neg不为"Covered by Model"的条目
    rti_neg_not_covered_df = df[df['Uncertainty_RTI_neg'] != 'Covered by Model']
    
    # 3. 同时满足两个条件的条目
    both_covered_df = df[(df['Uncertainty_RTI_pos'] == 'Covered by Model') & 
                         (df['Uncertainty_RTI_neg'] == 'Covered by Model')]
    
    # 不满足任一条件的条目
    neither_df = df[(df['Uncertainty_RTI_pos'] != 'Covered by Model') & 
                    (df['Uncertainty_RTI_neg'] != 'Covered by Model')]
    
    # 保存到新文件
    output_files = {
        'rti_pos_covered': '/home/data2/rhj/project/gnn/gnn-1/data/MMF-2/MMF-GNN_RTI_pos_Covered_by_Model.csv',
        'rti_pos_not_covered': '/home/data2/rhj/project/gnn/gnn-1/data/MMF-2/MMF-GNN_RTI_pos_Not_Covered_by_Model.csv',
        'rti_neg_covered': '/home/data2/rhj/project/gnn/gnn-1/data/MMF-2/MMF-GNN_RTI_neg_Covered_by_Model.csv',
        'rti_neg_not_covered': '/home/data2/rhj/project/gnn/gnn-1/data/MMF-2/MMF-GNN_RTI_neg_Not_Covered_by_Model.csv',
        'both': '/home/data2/rhj/project/gnn/gnn-1/data/MMF-2/MMF-GNN_both_RTI_Covered_by_Model.csv',
        'neither': '/home/data2/rhj/project/gnn/gnn-1/data/MMF-2/MMF-GNN_neither_RTI_Covered_by_Model.csv',
        'uncovered': '/home/data2/rhj/project/gnn/gnn-1/data/MMF-2/MMF-GNN_uncovered.csv'
    }
    
    rti_pos_covered_df.to_csv(output_files['rti_pos_covered'], index=False)
    rti_pos_not_covered_df.to_csv(output_files['rti_pos_not_covered'], index=False)
    rti_neg_covered_df.to_csv(output_files['rti_neg_covered'], index=False)
    rti_neg_not_covered_df.to_csv(output_files['rti_neg_not_covered'], index=False)
    both_covered_df.to_csv(output_files['both'], index=False)
    neither_df.to_csv(output_files['neither'], index=False)
    uncovered_df.to_csv(output_files['uncovered'], index=False)
    
    # 显示结果信息
    print(f"仅Uncertainty_RTI_pos为'Covered by Model'的数据集大小: {rti_pos_covered_df.shape}")
    print(f"仅Uncertainty_RTI_pos不为'Covered by Model'的数据集大小: {rti_pos_not_covered_df.shape}")
    print(f"仅Uncertainty_RTI_neg为'Covered by Model'的数据集大小: {rti_neg_covered_df.shape}")
    print(f"仅Uncertainty_RTI_neg不为'Covered by Model'的数据集大小: {rti_neg_not_covered_df.shape}")
    print(f"同时满足两个条件的数据集大小: {both_covered_df.shape}")
    print(f"两个条件都不满足的数据集大小: {neither_df.shape}")
    print(f"任意字段不被模型覆盖的数据集大小: {uncovered_df.shape}")
    
    print("\n文件保存信息:")
    for key, path in output_files.items():
        print(f"{key}: {path}")
    
    # 显示前几行以验证结果
    print("\n仅Uncertainty_RTI_pos为'Covered by Model'的前3行:")
    print(rti_pos_covered_df.head(3))
    
    print("\n仅Uncertainty_RTI_pos不为'Covered by Model'的前3行:")
    print(rti_pos_not_covered_df.head(3))
    
    print("\n仅Uncertainty_RTI_neg为'Covered by Model'的前3行:")
    print(rti_neg_covered_df.head(3))
    
    print("\n仅Uncertainty_RTI_neg不为'Covered by Model'的前3行:")
    print(rti_neg_not_covered_df.head(3))
    
    print("\n同时满足两个条件的前3行:")
    print(both_covered_df.head(3))
    
    print("\n两个条件都不满足的前3行:")
    print(neither_df.head(3))
    
    print("\n任意字段不被模型覆盖的前3行:")
    print(uncovered_df.head(3))
    
    return (rti_pos_covered_df, rti_pos_not_covered_df, 
            rti_neg_covered_df, rti_neg_not_covered_df, 
            both_covered_df, neither_df, uncovered_df)

if __name__ == "__main__":
    input_path = "/home/data2/rhj/project/gnn/gnn-1/data/MMF-2/MMF-GNN-valid-smiles.csv"
    
    try:
        result = filter_data_separate_and_combined(input_path)
    except FileNotFoundError:
        print(f"文件未找到，请检查路径: {input_path}")
    except Exception as e:
        print(f"处理数据时出错: {e}")