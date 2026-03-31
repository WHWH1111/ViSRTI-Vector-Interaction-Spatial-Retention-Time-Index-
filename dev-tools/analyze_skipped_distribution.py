import pandas as pd
from collections import Counter

def analyze_skipped_distribution():
    # 读取原始数据集
    original_df = pd.read_csv('/home/data2/rhj/project/gnn/gnn-1/data/MMF-2/MMF-GNN-valid-smiles.csv')
    
    # 读取跳过的分子数据
    skipped_df = pd.read_csv('/home/data2/rhj/project/gnn/gnn-1/predictions/neg-prediction_20251028-012019/MMF-GNN-valid-smiles_skipped_molecules.csv')
    
    # 创建一个字典来存储原始数据中的smiles及其对应的不确定性标签
    original_dict = {}
    for _, row in original_df.iterrows():
        smiles = row['SMILES']
        uncertainty_pos = row['Uncertainty_RTI_pos']
        uncertainty_neg = row['Uncertainty_RTI_neg']
        original_dict[smiles] = {
            'Uncertainty_RTI_pos': uncertainty_pos,
            'Uncertainty_RTI_neg': uncertainty_neg
        }
    
    # 收集跳过的分子在原始数据中的不确定性标签
    skipped_pos_labels = []
    skipped_neg_labels = []
    found_smiles = []
    not_found_smiles = []
    found_count = 0
    not_found_count = 0
    
    for _, row in skipped_df.iterrows():
        smiles = row['smiles']
        if smiles in original_dict:
            skipped_pos_labels.append(original_dict[smiles]['Uncertainty_RTI_pos'])
            skipped_neg_labels.append(original_dict[smiles]['Uncertainty_RTI_neg'])
            found_smiles.append(smiles)
            found_count += 1
        else:
            not_found_smiles.append(smiles)
            not_found_count += 1
    
    # 统计分布
    pos_distribution = Counter(skipped_pos_labels)
    neg_distribution = Counter(skipped_neg_labels)
    
    # 打印结果
    print(f"总共找到 {found_count} 个跳过的分子在原始数据集中")
    print(f"未找到 {not_found_count} 个跳过的分子在原始数据集中")
    
    # 显示未找到的前5个SMILES
    print("\n未找到的前5个跳过的分子SMILES:")
    for i, smiles in enumerate(not_found_smiles[:5]):
        print(f"  {i+1}. {smiles}")
    
    print("\nUncertainty_RTI_pos 分布:")
    for label, count in pos_distribution.items():
        print(f"  {label}: {count}")
    
    print("\nUncertainty_RTI_neg 分布:")
    for label, count in neg_distribution.items():
        print(f"  {label}: {count}")
    
    # 计算百分比
    print("\nUncertainty_RTI_pos 百分比分布:")
    total = len(skipped_pos_labels)
    for label, count in pos_distribution.items():
        percentage = (count / total) * 100
        print(f"  {label}: {count}/{total} ({percentage:.2f}%)")
    
    print("\nUncertainty_RTI_neg 百分比分布:")
    total = len(skipped_neg_labels)
    for label, count in neg_distribution.items():
        percentage = (count / total) * 100
        print(f"  {label}: {count}/{total} ({percentage:.2f}%)")

if __name__ == "__main__":
    analyze_skipped_distribution()