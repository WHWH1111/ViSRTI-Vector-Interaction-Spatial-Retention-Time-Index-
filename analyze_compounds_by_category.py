import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import warnings
import os
import argparse

warnings.filterwarnings('ignore')

# 十大分类（固定）
MAJOR_CATEGORIES = [
    'Aromatic',                 # 芳香化合物（结构）
    'Aliphatic',                # 脂肪族化合物（结构）
    'Nitrogen_containing',      # 含氮化合物（结构）
    'Oxygen_containing',        # 含氧化合物（结构）
    'Sulfur_containing',        # 含硫化合物（结构）
    'Phosphorus_containing',    # 含磷化合物（结构）
    'Halogen_containing',       # 含卤化合物（结构）
    'High_hydrophobic',         # 高疏水性（性质）
    'Medium_hydrophobic',       # 中等疏水性（性质）
    'Hydrophilic'               # 亲水性（性质）
]

# 哪些类别需要子类划分
CATEGORIES_WITH_SUBCLASS = {
    'Aromatic': True,
    'Aliphatic': True,
    'Nitrogen_containing': True,
    'Oxygen_containing': True,
    'Sulfur_containing': True,
    'Phosphorus_containing': True,
    'Halogen_containing': True,
    'High_hydrophobic': False,   # 性质类不需要子类
    'Medium_hydrophobic': False, # 性质类不需要子类
    'Hydrophilic': False         # 性质类不需要子类
}

def detect_functional_groups(smiles):
    """识别分子中的官能团"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    
    # 基本检测
    func_groups = {
        'Aromatic': mol.HasSubstructMatch(Chem.MolFromSmarts('a')),
    }
    
    # 计算芳香环数
    aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    func_groups['Polycyclic_Aromatic'] = aromatic_rings >= 2
    func_groups['AromaticRings'] = aromatic_rings
    
    # 氮相关
    func_groups['Amine_Primary'] = mol.HasSubstructMatch(Chem.MolFromSmarts('[NH2;!$(NC=[O,S,N])]'))
    func_groups['Amine_Secondary'] = mol.HasSubstructMatch(Chem.MolFromSmarts('[NH1;!$(N(C)(C)C=[O,S,N])]'))
    func_groups['Amine_Tertiary'] = mol.HasSubstructMatch(Chem.MolFromSmarts('[NH0;!$(N(C)(C)C=[O,S,N])]'))
    func_groups['Amide'] = mol.HasSubstructMatch(Chem.MolFromSmarts('C(=O)[NH]'))
    func_groups['Nitro'] = mol.HasSubstructMatch(Chem.MolFromSmarts('[N+](=O)[O-]'))
    func_groups['Nitrile'] = mol.HasSubstructMatch(Chem.MolFromSmarts('C#N'))
    
    # 氧相关
    func_groups['Alcohol_Primary'] = mol.HasSubstructMatch(Chem.MolFromSmarts('[CH2][OH]'))
    func_groups['Alcohol_Secondary'] = mol.HasSubstructMatch(Chem.MolFromSmarts('[CH]([CH3])[OH]'))
    func_groups['Alcohol_Tertiary'] = mol.HasSubstructMatch(Chem.MolFromSmarts('C(C)(C)[OH]'))
    func_groups['Phenol'] = mol.HasSubstructMatch(Chem.MolFromSmarts('c[OH]'))
    func_groups['Carboxylic_Acid'] = mol.HasSubstructMatch(Chem.MolFromSmarts('C(=O)[OH]'))
    func_groups['Aldehyde'] = mol.HasSubstructMatch(Chem.MolFromSmarts('[CH](=O)'))
    func_groups['Ketone'] = mol.HasSubstructMatch(Chem.MolFromSmarts('C(=O)C'))
    func_groups['Ester'] = mol.HasSubstructMatch(Chem.MolFromSmarts('C(=O)[O][C]'))
    func_groups['Ether'] = mol.HasSubstructMatch(Chem.MolFromSmarts('COC'))
    
    # 硫相关
    func_groups['Thiol'] = mol.HasSubstructMatch(Chem.MolFromSmarts('[SH]'))
    func_groups['Sulfide'] = mol.HasSubstructMatch(Chem.MolFromSmarts('CSC'))
    func_groups['Sulfoxide'] = mol.HasSubstructMatch(Chem.MolFromSmarts('CS(=O)C'))
    func_groups['Sulfone'] = mol.HasSubstructMatch(Chem.MolFromSmarts('CS(=O)(=O)C'))
    func_groups['Sulfonic_Acid'] = mol.HasSubstructMatch(Chem.MolFromSmarts('S(=O)(=O)[OH]'))
    
    # 磷相关
    func_groups['Phosphate_Ester'] = mol.HasSubstructMatch(Chem.MolFromSmarts('P(=O)([O])[O]'))
    
    # 卤素
    smiles_str = Chem.MolToSmiles(mol)
    func_groups['Fluorine'] = 'F' in smiles_str
    func_groups['Chlorine'] = 'Cl' in smiles_str
    func_groups['Bromine'] = 'Br' in smiles_str
    func_groups['Iodine'] = 'I' in smiles_str
    
    # 其他结构
    func_groups['Alkene'] = mol.HasSubstructMatch(Chem.MolFromSmarts('C=C'))
    func_groups['Alkyne'] = mol.HasSubstructMatch(Chem.MolFromSmarts('C#C'))
    func_groups['Cycloalkane'] = mol.HasSubstructMatch(Chem.MolFromSmarts('C1CCCC1'))
    
    # 杂原子计数
    func_groups['Nitrogen_Count'] = len([a for a in mol.GetAtoms() if a.GetSymbol() == 'N'])
    func_groups['Oxygen_Count'] = len([a for a in mol.GetAtoms() if a.GetSymbol() == 'O'])
    func_groups['Sulfur_Count'] = len([a for a in mol.GetAtoms() if a.GetSymbol() == 'S'])
    func_groups['Phosphorus_Count'] = len([a for a in mol.GetAtoms() if a.GetSymbol() == 'P'])
    func_groups['Halogen_Count'] = len([a for a in mol.GetAtoms() if a.GetSymbol() in ['F', 'Cl', 'Br', 'I']])
    
    # 检测芳香杂环（芳香环中有非碳原子）
    func_groups['Heterocyclic'] = False
    if func_groups['Aromatic']:
        for atom in mol.GetAtoms():
            if atom.IsInRing() and atom.GetIsAromatic() and atom.GetSymbol() != 'C':
                func_groups['Heterocyclic'] = True
                break
    
    return func_groups

def calculate_basic_descriptors(smiles):
    """计算基本的分子描述符"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    
    descriptors = {
        'MW': Descriptors.ExactMolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'HBD': Descriptors.NumHDonors(mol),
        'HBA': Descriptors.NumHAcceptors(mol),
        'RotatableBonds': Descriptors.NumRotatableBonds(mol),
        'TPSA': Descriptors.TPSA(mol),
        'RingCount': Descriptors.RingCount(mol),
        'AromaticRings': rdMolDescriptors.CalcNumAromaticRings(mol),
    }
    return descriptors

def categorize_by_hydrophobicity(logp):
    """根据logP值分类亲疏水性"""
    if pd.isna(logp):
        return 'Unknown'
    elif logp <= 2:
        return 'Hydrophilic'
    elif logp <= 4:
        return 'Medium_hydrophobic'
    else:
        return 'High_hydrophobic'

def belongs_to_category(func_groups, category):
    """判断分子是否属于某个类别"""
    if category == 'Aromatic':
        return func_groups.get('Aromatic', False)
    
    elif category == 'Aliphatic':
        return not func_groups.get('Aromatic', False)
    
    elif category == 'Nitrogen_containing':
        return any([
            func_groups.get('Nitrogen_Count', 0) > 0,
            func_groups.get('Amine_Primary', False),
            func_groups.get('Amine_Secondary', False),
            func_groups.get('Amine_Tertiary', False),
            func_groups.get('Amide', False),
            func_groups.get('Nitro', False),
            func_groups.get('Nitrile', False)
        ])
    
    elif category == 'Oxygen_containing':
        return any([
            func_groups.get('Oxygen_Count', 0) > 0,
            func_groups.get('Alcohol_Primary', False),
            func_groups.get('Alcohol_Secondary', False),
            func_groups.get('Alcohol_Tertiary', False),
            func_groups.get('Phenol', False),
            func_groups.get('Carboxylic_Acid', False),
            func_groups.get('Aldehyde', False),
            func_groups.get('Ketone', False),
            func_groups.get('Ester', False),
            func_groups.get('Ether', False)
        ])
    
    elif category == 'Sulfur_containing':
        return any([
            func_groups.get('Sulfur_Count', 0) > 0,
            func_groups.get('Thiol', False),
            func_groups.get('Sulfide', False),
            func_groups.get('Sulfoxide', False),
            func_groups.get('Sulfone', False),
            func_groups.get('Sulfonic_Acid', False)
        ])
    
    elif category == 'Phosphorus_containing':
        return any([
            func_groups.get('Phosphorus_Count', 0) > 0,
            func_groups.get('Phosphate_Ester', False)
        ])
    
    elif category == 'Halogen_containing':
        return any([
            func_groups.get('Halogen_Count', 0) > 0,
            func_groups.get('Fluorine', False),
            func_groups.get('Chlorine', False),
            func_groups.get('Bromine', False),
            func_groups.get('Iodine', False)
        ])
    
    # 亲疏水性类别通过LogP判断，这里返回False
    elif category in ['High_hydrophobic', 'Medium_hydrophobic', 'Hydrophilic']:
        return False
    
    return False

def get_subcategories(func_groups, category):
    """获取指定类别的子类标签"""
    subcategories = []
    
    if category == 'Aromatic':
        aromatic_rings = func_groups.get('AromaticRings', 0)
        if aromatic_rings >= 2:
            subcategories.append('Polycyclic_Aromatic')
            if aromatic_rings >= 4:
                subcategories.append('PAH')
        else:
            subcategories.append('Monocyclic_Aromatic')
        
        # 检测芳香杂环
        if func_groups.get('Heterocyclic', False):
            subcategories.append('Heterocyclic_Aromatic')
        
        if func_groups.get('Phenol', False):
            subcategories.append('Aromatic_Phenol')
        if func_groups.get('Carboxylic_Acid', False):
            subcategories.append('Aromatic_Carboxylic_Acid')
        if func_groups.get('Amine_Primary', False) or func_groups.get('Amine_Secondary', False) or func_groups.get('Amine_Tertiary', False):
            subcategories.append('Aromatic_Amine')
        if func_groups.get('Ketone', False):
            subcategories.append('Aromatic_Ketone')
        if func_groups.get('Aldehyde', False):
            subcategories.append('Aromatic_Aldehyde')
        if func_groups.get('Ether', False):
            subcategories.append('Aromatic_Ether')
        if func_groups.get('Nitro', False):
            subcategories.append('Aromatic_Nitro')
    
    elif category == 'Aliphatic':
        if func_groups.get('Alkene', False):
            subcategories.append('Alkene')
        if func_groups.get('Alkyne', False):
            subcategories.append('Alkyne')
        if func_groups.get('Cycloalkane', False):
            subcategories.append('Cycloalkane')
        
        if func_groups.get('Alcohol_Primary', False):
            subcategories.append('Primary_Alcohol')
        elif func_groups.get('Alcohol_Secondary', False):
            subcategories.append('Secondary_Alcohol')
        elif func_groups.get('Alcohol_Tertiary', False):
            subcategories.append('Tertiary_Alcohol')
        
        if func_groups.get('Carboxylic_Acid', False):
            subcategories.append('Carboxylic_Acid')
        if func_groups.get('Ester', False):
            subcategories.append('Ester')
        if func_groups.get('Ether', False):
            subcategories.append('Ether')
        if func_groups.get('Amine_Primary', False) or func_groups.get('Amine_Secondary', False) or func_groups.get('Amine_Tertiary', False):
            subcategories.append('Amine')
        if func_groups.get('Ketone', False):
            subcategories.append('Ketone')
        if func_groups.get('Aldehyde', False):
            subcategories.append('Aldehyde')
    
    elif category == 'Nitrogen_containing':
        if func_groups.get('Amine_Primary', False):
            subcategories.append('Primary_Amine')
        if func_groups.get('Amine_Secondary', False):
            subcategories.append('Secondary_Amine')
        if func_groups.get('Amine_Tertiary', False):
            subcategories.append('Tertiary_Amine')
        if func_groups.get('Amide', False):
            subcategories.append('Amide')
        if func_groups.get('Nitro', False):
            subcategories.append('Nitro')
        if func_groups.get('Nitrile', False):
            subcategories.append('Nitrile')
    
    elif category == 'Oxygen_containing':
        if func_groups.get('Alcohol_Primary', False):
            subcategories.append('Primary_Alcohol')
        if func_groups.get('Alcohol_Secondary', False):
            subcategories.append('Secondary_Alcohol')
        if func_groups.get('Alcohol_Tertiary', False):
            subcategories.append('Tertiary_Alcohol')
        if func_groups.get('Phenol', False):
            subcategories.append('Phenol')
        if func_groups.get('Carboxylic_Acid', False):
            subcategories.append('Carboxylic_Acid')
        if func_groups.get('Aldehyde', False):
            subcategories.append('Aldehyde')
        if func_groups.get('Ketone', False):
            subcategories.append('Ketone')
        if func_groups.get('Ester', False):
            subcategories.append('Ester')
        if func_groups.get('Ether', False):
            subcategories.append('Ether')
    
    elif category == 'Sulfur_containing':
        if func_groups.get('Thiol', False):
            subcategories.append('Thiol')
        if func_groups.get('Sulfide', False):
            subcategories.append('Sulfide')
        if func_groups.get('Sulfoxide', False):
            subcategories.append('Sulfoxide')
        if func_groups.get('Sulfone', False):
            subcategories.append('Sulfone')
        if func_groups.get('Sulfonic_Acid', False):
            subcategories.append('Sulfonic_Acid')
    
    elif category == 'Phosphorus_containing':
        if func_groups.get('Phosphate_Ester', False):
            subcategories.append('Phosphate_Ester')
    
    elif category == 'Halogen_containing':
        if func_groups.get('Fluorine', False):
            subcategories.append('Fluorine')
        if func_groups.get('Chlorine', False):
            subcategories.append('Chlorine')
        if func_groups.get('Bromine', False):
            subcategories.append('Bromine')
        if func_groups.get('Iodine', False):
            subcategories.append('Iodine')
    
    return subcategories

def generate_category_tag(func_groups, descriptors):
    """生成分类标签，格式：类别(子类1,子类2,...);类别(子类...);类别"""
    category_tags = []
    
    # 处理结构类别
    for category in ['Aromatic', 'Aliphatic', 'Nitrogen_containing', 'Oxygen_containing',
                    'Sulfur_containing', 'Phosphorus_containing', 'Halogen_containing']:
        
        if belongs_to_category(func_groups, category):
            if CATEGORIES_WITH_SUBCLASS[category]:
                subcategories = get_subcategories(func_groups, category)
                if subcategories:
                    # 格式：类别(子类1,子类2,...)
                    category_tags.append(f"{category}({','.join(subcategories)})")
                else:
                    # 如果没有检测到具体子类，只添加主类
                    category_tags.append(category)
            else:
                category_tags.append(category)
    
    # 处理性质类别（根据LogP）
    logp = descriptors.get('LogP', np.nan)
    if not pd.isna(logp):
        if logp <= 2:
            category_tags.append('Hydrophilic')
        elif logp <= 4:
            category_tags.append('Medium_hydrophobic')
        else:
            category_tags.append('High_hydrophobic')
    
    # 用分号连接所有类别
    return ';'.join(category_tags)

def classify_compounds(df):
    """主分类函数"""
    # 新增列
    df['category_tag'] = ''
    df['LogP'] = np.nan
    df['aromatic_rings'] = 0
    df['is_heterocyclic'] = False
    
    # 统计
    category_counts = {cat: 0 for cat in MAJOR_CATEGORIES}
    subcategory_counts = {}
    
    for idx, row in df.iterrows():
        # 获取SMILES
        smiles = row['SMILES_pos'] if pd.notna(row['SMILES_pos']) else row['SMILES_neg']
        if pd.isna(smiles):
            continue
        
        # 计算描述符和官能团
        descriptors = calculate_basic_descriptors(smiles)
        func_groups = detect_functional_groups(smiles)
        
        # 存储基本信息
        logp = descriptors.get('LogP', np.nan)
        df.at[idx, 'LogP'] = logp
        df.at[idx, 'aromatic_rings'] = func_groups.get('AromaticRings', 0)
        df.at[idx, 'is_heterocyclic'] = func_groups.get('Heterocyclic', False)
        
        # 生成分类标签
        category_tag = generate_category_tag(func_groups, descriptors)
        df.at[idx, 'category_tag'] = category_tag
        
        # 统计每个类别的出现次数
        for category in MAJOR_CATEGORIES:
            if category in category_tag:
                category_counts[category] += 1
        
        # 统计子类分布
        tag_parts = category_tag.split(';')
        for tag_part in tag_parts:
            if '(' in tag_part:  # 包含子类的标签
                main_category = tag_part.split('(')[0]
                subcategories = tag_part.split('(')[1].rstrip(')').split(',')
                
                if main_category not in subcategory_counts:
                    subcategory_counts[main_category] = {}
                
                for subcat in subcategories:
                    if subcat not in subcategory_counts[main_category]:
                        subcategory_counts[main_category][subcat] = 0
                    subcategory_counts[main_category][subcat] += 1
    
    return df, category_counts, subcategory_counts

def analyze_results(df, category_counts, subcategory_counts):
    """分析分类结果"""
    analysis = {}
    
    analysis['category_counts'] = category_counts
    
    # 计算平均类别数
    df['num_categories'] = df['category_tag'].apply(lambda x: len(x.split(';')) if x else 0)
    avg_categories = df['num_categories'].mean()
    analysis['avg_categories_per_compound'] = avg_categories
    
    # LogP分布
    analysis['logp_stats'] = {
        'min': df['LogP'].min(),
        'max': df['LogP'].max(),
        'mean': df['LogP'].mean(),
        'median': df['LogP'].median()
    }
    
    # 子类统计
    analysis['subcategory_counts'] = subcategory_counts
    
    # 最常见的分类标签
    tag_counts = df['category_tag'].value_counts()
    analysis['common_tags'] = tag_counts.head(10).to_dict()
    
    return analysis

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='化合物十大分类分析')
    parser.add_argument('--input', type=str, default='predictions/visnet-v2-5fold/complete_dataset.csv',
                        help='输入CSV文件路径')
    parser.add_argument('--output', type=str, default='predictions/visnet-v2-5fold/categorized_dataset.csv',
                        help='输出CSV文件路径')
    parser.add_argument('--summary', type=str, default='predictions/visnet-v2-5fold/category_summary.txt',
                        help='汇总统计文件路径')
    args = parser.parse_args()
    
    # 读取数据
    input_file = args.input
    if not os.path.exists(input_file):
        print(f"文件 {input_file} 不存在")
        return
    
    # 尝试不同编码
    encodings = ['utf-8', 'gbk', 'latin1']
    df = None
    for encoding in encodings:
        try:
            df = pd.read_csv(input_file, encoding=encoding)
            print(f"成功使用 {encoding} 编码读取文件")
            break
        except UnicodeDecodeError:
            continue
    
    if df is None:
        print("无法读取文件")
        return
    
    print(f"总共有 {len(df)} 个化合物")
    
    # 进行分类
    df, category_counts, subcategory_counts = classify_compounds(df)
    
    # 分析结果
    analysis = analyze_results(df, category_counts, subcategory_counts)
    
    # 打印结果
    print("\n=== 十大分类分布 ===")
    total_assignments = 0
    for category in MAJOR_CATEGORIES:
        count = category_counts[category]
        total_assignments += count
        percentage = count / len(df) * 100
        print(f"{category}: {count} 个化合物 ({percentage:.1f}%)")
    
    print(f"\n总类别分配数: {total_assignments}")
    print(f"平均每个化合物属于 {analysis['avg_categories_per_compound']:.2f} 个类别")
    
    print("\n=== LogP统计 ===")
    stats = analysis['logp_stats']
    print(f"最小值: {stats['min']:.2f}")
    print(f"最大值: {stats['max']:.2f}")
    print(f"平均值: {stats['mean']:.2f}")
    print(f"中位数: {stats['median']:.2f}")
    
    print("\n=== 主要子类分布 ===")
    for category, subcats in subcategory_counts.items():
        if subcats:
            print(f"\n{category}:")
            sorted_subcats = sorted(subcats.items(), key=lambda x: x[1], reverse=True)
            for subcat, count in sorted_subcats[:5]:  # 显示前5个最常见的子类
                print(f"  {subcat}: {count}")
    
    print("\n=== 最常见的分类标签（前10） ===")
    for tag, count in analysis['common_tags'].items():
        print(f"{tag}: {count}")
    
    # 保存结果
    output_file = args.output
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n分类结果已保存到 {output_file}")
    
    # 保存汇总统计
    summary_file = args.summary
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=== 化合物分类汇总 ===\n\n")
        f.write(f"总化合物数: {len(df)}\n\n")
        
        f.write("十大分类分布:\n")
        for category in MAJOR_CATEGORIES:
            count = category_counts[category]
            percentage = count / len(df) * 100
            f.write(f"  {category}: {count} ({percentage:.1f}%)\n")
        
        f.write(f"\n总类别分配数: {total_assignments}\n")
        f.write(f"平均每个化合物属于 {analysis['avg_categories_per_compound']:.2f} 个类别\n\n")
        
        f.write("LogP统计:\n")
        for stat_name, stat_value in analysis['logp_stats'].items():
            f.write(f"  {stat_name}: {stat_value:.2f}\n")
        
        f.write("\n子类分布:\n")
        for category, subcats in subcategory_counts.items():
            if subcats:
                f.write(f"\n  {category}:\n")
                sorted_subcats = sorted(subcats.items(), key=lambda x: x[1], reverse=True)
                for subcat, count in sorted_subcats[:10]:
                    percentage = count / category_counts[category] * 100 if category_counts[category] > 0 else 0
                    f.write(f"    {subcat}: {count} ({percentage:.1f}%)\n")
        
        f.write("\n最常见的分类标签（前20）:\n")
        tag_counts = df['category_tag'].value_counts()
        for tag, count in tag_counts.head(20).items():
            percentage = count / len(df) * 100
            f.write(f"  {tag}: {count} ({percentage:.1f}%)\n")
    
    print(f"汇总统计已保存到 {summary_file}")
    
    # 显示前10个化合物的分类结果
    print("\n=== 前10个化合物的分类结果 ===")
    for i in range(min(10, len(df))):
        row = df.iloc[i]
        name = str(row.get('Name_neg', '') or row.get('name', ''))[:50]
        print(f"\n{i+1}. {row.get('Norman_SusDat_ID', f'Compound_{i+1}')} ({name}...)")
        print(f"   分类标签: {row['category_tag']}")
        print(f"   LogP: {row['LogP']:.2f}, 芳香环数: {row['aromatic_rings']}")

if __name__ == "__main__":
    main()