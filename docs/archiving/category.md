## **一、RDKit用于亚类识别的核心指令**

首先，你需要在Python中安装并导入RDKit：

```python
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors
from rdkit.Chem import AllChem
```

### **1. 基础的分子描述符计算**

```python
def calculate_basic_descriptors(smiles):
    """计算基本的分子描述符"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    descriptors = {
        'MW': Descriptors.ExactMolWt(mol),  # 分子量
        'LogP': Descriptors.MolLogP(mol),    # 计算logP
        'HBD': Descriptors.NumHDonors(mol),  # 氢键供体数
        'HBA': Descriptors.NumHAcceptors(mol), # 氢键受体数
        'RotatableBonds': Descriptors.NumRotatableBonds(mol), # 可旋转键数
        'TPSA': Descriptors.TPSA(mol),      # 极性表面积
        'RingCount': Descriptors.RingCount(mol), # 环数
        'AromaticRings': rdMolDescriptors.CalcNumAromaticRings(mol), # 芳香环数
    }
    return descriptors
```

### **2. 官能团亚类识别函数**

以下是识别各种官能团亚类的核心函数：

```python
def detect_functional_groups(smiles):
    """识别分子中的官能团亚类"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # 初始化官能团字典
    func_groups = {
        # 芳香族相关
        'Aromatic': mol.HasSubstructMatch(Chem.MolFromSmarts('a')),  # 是否有芳香原子
        'Polycyclic_Aromatic': False,  # 多环芳烃（稍后判断）
        
        # 氮相关
        'Amine_Primary': mol.HasSubstructMatch(Chem.MolFromSmarts('[NH2]C')),  # 伯胺
        'Amine_Secondary': mol.HasSubstructMatch(Chem.MolFromSmarts('[NH](C)C')),  # 仲胺
        'Amine_Tertiary': mol.HasSubstructMatch(Chem.MolFromSmarts('[N](C)(C)C')),  # 叔胺
        'Amide': mol.HasSubstructMatch(Chem.MolFromSmarts('C(=O)N')),  # 酰胺
        'Nitro': mol.HasSubstructMatch(Chem.MolFromSmarts('[N+](=O)[O-]')),  # 硝基
        'Nitrile': mol.HasSubstructMatch(Chem.MolFromSmarts('C#N')),  # 腈
        'Amino_Acid': mol.HasSubstructMatch(Chem.MolFromSmarts('[NH2]CC(=O)[OH]')),  # 氨基酸
        
        # 氧相关
        'Alcohol_Primary': mol.HasSubstructMatch(Chem.MolFromSmarts('[CH2][OH]')),  # 伯醇
        'Alcohol_Secondary': mol.HasSubstructMatch(Chem.MolFromSmarts('[CH]([CH3])[OH]')),  # 仲醇
        'Alcohol_Tertiary': mol.HasSubstructMatch(Chem.MolFromSmarts('C(C)(C)[OH]')),  # 叔醇
        'Phenol': mol.HasSubstructMatch(Chem.MolFromSmarts('c[OH]')),  # 酚
        'Carboxylic_Acid': mol.HasSubstructMatch(Chem.MolFromSmarts('C(=O)[OH]')),  # 羧酸
        'Aldehyde': mol.HasSubstructMatch(Chem.MolFromSmarts('[CH](=O)')),  # 醛
        'Ketone': mol.HasSubstructMatch(Chem.MolFromSmarts('C(=O)C')),  # 酮
        'Ester': mol.HasSubstructMatch(Chem.MolFromSmarts('C(=O)[O][C]')),  # 酯
        'Ether': mol.HasSubstructMatch(Chem.MolFromSmarts('COC')),  # 醚
        
        # 硫/磷相关
        'Thiol': mol.HasSubstructMatch(Chem.MolFromSmarts('[SH]')),  # 硫醇
        'Sulfide': mol.HasSubstructMatch(Chem.MolFromSmarts('CSC')),  # 硫醚
        'Sulfoxide': mol.HasSubstructMatch(Chem.MolFromSmarts('CS(=O)C')),  # 亚砜
        'Sulfone': mol.HasSubstructMatch(Chem.MolFromSmarts('CS(=O)(=O)C')),  # 砜
        'Sulfonic_Acid': mol.HasSubstructMatch(Chem.MolFromSmarts('S(=O)(=O)[OH]')),  # 磺酸
        'Phosphate_Ester': mol.HasSubstructMatch(Chem.MolFromSmarts('P(=O)([O])[O]')),  # 磷酸酯
        
        # 卤素相关
        'Fluorine': 'F' in Chem.MolToSmiles(mol),  # 含氟
        'Chlorine': 'Cl' in Chem.MolToSmiles(mol),  # 含氯
        'Bromine': 'Br' in Chem.MolToSmiles(mol),  # 含溴
        'Iodine': 'I' in Chem.MolToSmiles(mol),  # 含碘
        'Halogen_Count': len([atom for atom in mol.GetAtoms() if atom.GetSymbol() in ['F', 'Cl', 'Br', 'I']]),  # 卤素原子总数
        
        # 其他
        'Alkane': mol.HasSubstructMatch(Chem.MolFromSmarts('CC')),  # 烷烃特征
        'Alkene': mol.HasSubstructMatch(Chem.MolFromSmarts('C=C')),  # 烯烃
        'Alkyne': mol.HasSubstructMatch(Chem.MolFromSmarts('C#C')),  # 炔烃
    }
    
    # 判断多环芳烃：芳香环数 >= 2
    func_groups['Polycyclic_Aromatic'] = (rdMolDescriptors.CalcNumAromaticRings(mol) >= 2)
    
    return func_groups
```

---

## **二、每个大类下的亚类覆盖策略**

### **1. 芳香化合物 (Aromatic Compounds)**
需要覆盖的亚类：
- **单环芳烃**：苯及其简单取代物
- **多环芳烃(PAHs)**：萘、菲、芘等
- **杂环芳烃**：含N/O/S的芳香环（如吡啶、呋喃、噻吩）
- **芳香醇/酚**：芳香环上连-OH
- **芳香酸**：芳香环上连-COOH
- **芳香胺**：芳香环上连-NH₂
- **芳香酮/醛**：芳香环上连-C=O

**选择策略**：
```python
def select_aromatic_compounds(df):
    """从数据框中选择芳香化合物，确保亚类覆盖"""
    aromatic_df = df[df['is_aromatic'] == True]
    
    selected = []
    
    # 1. 单环芳烃 (选择1-2个)
    mono_aromatic = aromatic_df[aromatic_df['aromatic_rings'] == 1]
    selected.extend(mono_aromatic.sample(2).index.tolist())
    
    # 2. 多环芳烃 (选择2-3个)
    poly_aromatic = aromatic_df[aromatic_df['aromatic_rings'] >= 2]
    selected.extend(poly_aromatic.sample(3).index.tolist())
    
    # 3. 芳香醇/酚 (选择1个)
    phenols = aromatic_df[aromatic_df['Phenol'] == True]
    selected.extend(phenols.sample(1).index.tolist())
    
    # 4. 芳香酸 (选择1个)
    aromatic_acids = aromatic_df[aromatic_df['Carboxylic_Acid'] == True]
    selected.extend(aromatic_acids.sample(1).index.tolist())
    
    # 5. 芳香胺 (选择1个)
    aromatic_amines = aromatic_df[aromatic_df['Amine_Primary'] == True]
    selected.extend(aromatic_amines.sample(1).index.tolist())
    
    return selected
```

### **2. 脂肪族化合物 (Aliphatic Compounds)**
需要覆盖的亚类：
- **直链烷烃**：正己烷、正癸烷等
- **支链烷烃**：异辛烷等
- **环烷烃**：环己烷、环戊烷
- **烯烃**：含C=C双键
- **炔烃**：含C≡C三键
- **脂肪醇**：乙醇、正丁醇
- **脂肪羧酸**：乙酸、丁酸
- **脂肪胺**：乙胺、丁胺
- **酯类**：乙酸乙酯
- **醚类**：乙醚

**选择策略**：
```python
def select_aliphatic_compounds(df):
    """选择脂肪族化合物，确保亚类覆盖"""
    # 先排除芳香化合物
    aliphatic_df = df[df['is_aromatic'] == False]
    
    selected = []
    
    # 按亚类选择
    subcategories = {
        'Alkane': '直链/支链烷烃',
        'Cycloalkane': '环烷烃',
        'Alkene': '烯烃',
        'Alkyne': '炔烃',
        'Alcohol': '脂肪醇',
        'Carboxylic_Acid': '脂肪羧酸',
        'Ester': '酯类',
        'Ether': '醚类',
        'Amine': '脂肪胺'
    }
    
    for subcat, desc in subcategories.items():
        if subcat in aliphatic_df.columns:
            candidates = aliphatic_df[aliphatic_df[subcat] == True]
            if len(candidates) > 0:
                selected.extend(candidates.sample(1).index.tolist())
    
    return selected
```

### **3. 含氮化合物 (Nitrogen-containing Compounds)**
需要覆盖的亚类：
- **伯胺** (-NH₂)
- **仲胺** (-NH-)
- **叔胺** (-N<)
- **酰胺** (-CONH-)
- **硝基化合物** (-NO₂)
- **腈类** (-C≡N)
- **氨基酸**
- **含氮杂环**：吡啶、吡咯、嘧啶等

### **4. 含氧化合物 (Oxygen-containing Compounds)**
需要覆盖的亚类：
- **伯醇** (-CH₂OH)
- **仲醇** (>CHOH)
- **叔醇** (>C-OH)
- **酚** (芳香-OH)
- **羧酸** (-COOH)
- **醛** (-CHO)
- **酮** (>C=O)
- **酯** (-COO-)
- **醚** (-O-)

### **5. 含硫/磷化合物 (Sulfur/Phosphorus-containing Compounds)**
需要覆盖的亚类：
- **硫醇/硫酚** (-SH)
- **硫醚** (-S-)
- **亚砜** (>S=O)
- **砜** (>SO₂)
- **磺酸** (-SO₃H)
- **磷酸酯** (>P=O(OR)₃)

### **6. 含卤化合物 (Halogen-containing Compounds)**
需要覆盖的亚类：
- **氟代物**
- **氯代物**
- **溴代物**
- **碘代物**
- **单卤代 vs 多卤代**
- **脂肪族卤代 vs 芳香族卤代**

---

## **三、完整的自动化选择脚本**

这里是一个完整的脚本，可以自动从你的数据集中选择覆盖各个亚类的化合物：

```python
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import warnings
warnings.filterwarnings('ignore')

class CompoundSelector:
    def __init__(self, data_path):
        """初始化选择器，加载数据"""
        self.df = pd.read_csv(data_path)
        self.selected_indices = []
        
    def preprocess_data(self):
        """预处理数据，计算所有描述符和官能团"""
        print("正在计算分子描述符和官能团...")
        
        # 计算基础描述符
        self.df['MW'] = self.df['SMILES'].apply(lambda x: Descriptors.ExactMolWt(Chem.MolFromSmiles(x)) if Chem.MolFromSmiles(x) else np.nan)
        self.df['LogP'] = self.df['SMILES'].apply(lambda x: Descriptors.MolLogP(Chem.MolFromSmiles(x)) if Chem.MolFromSmiles(x) else np.nan)
        self.df['HBD'] = self.df['SMILES'].apply(lambda x: Descriptors.NumHDonors(Chem.MolFromSmiles(x)) if Chem.MolFromSmiles(x) else np.nan)
        self.df['HBA'] = self.df['SMILES'].apply(lambda x: Descriptors.NumHAcceptors(Chem.MolFromSmiles(x)) if Chem.MolFromSmiles(x) else np.nan)
        self.df['TPSA'] = self.df['SMILES'].apply(lambda x: Descriptors.TPSA(Chem.MolFromSmiles(x)) if Chem.MolFromSmiles(x) else np.nan)
        
        # 判断芳香性
        self.df['is_aromatic'] = self.df['SMILES'].apply(lambda x: Chem.MolFromSmiles(x).HasSubstructMatch(Chem.MolFromSmarts('a')) if Chem.MolFromSmiles(x) else False)
        self.df['aromatic_rings'] = self.df['SMILES'].apply(lambda x: rdMolDescriptors.CalcNumAromaticRings(Chem.MolFromSmiles(x)) if Chem.MolFromSmiles(x) else 0)
        
        # 判断各官能团
        self.df['has_amine'] = self.df['SMILES'].apply(self.has_amine)
        self.df['has_amide'] = self.df['SMILES'].apply(self.has_amide)
        self.df['has_nitro'] = self.df['SMILES'].apply(self.has_nitro)
        self.df['has_alcohol'] = self.df['SMILES'].apply(self.has_alcohol)
        self.df['has_carboxylic_acid'] = self.df['SMILES'].apply(self.has_carboxylic_acid)
        self.df['has_ketone'] = self.df['SMILES'].apply(self.has_ketone)
        self.df['has_ester'] = self.df['SMILES'].apply(self.has_ester)
        self.df['has_halogen'] = self.df['SMILES'].apply(self.has_halogen)
        self.df['has_sulfur'] = self.df['SMILES'].apply(self.has_sulfur)
        self.df['has_phosphorus'] = self.df['SMILES'].apply(self.has_phosphorus)
        
        print(f"数据预处理完成，共处理{len(self.df)}个化合物")
        
    def has_amine(self, smiles):
        """判断是否有胺基"""
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return False
        return mol.HasSubstructMatch(Chem.MolFromSmarts('[N;!H0]'))
    
    def has_amide(self, smiles):
        """判断是否有酰胺"""
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return False
        return mol.HasSubstructMatch(Chem.MolFromSmarts('C(=O)N'))
    
    # 其他has_*函数类似定义...
    
    def categorize_compounds(self):
        """将化合物分配到十大类别"""
        categories = {
            'Aromatic': [],
            'Aliphatic': [],
            'Nitrogen': [],
            'Oxygen': [],
            'Sulfur_Phosphorus': [],
            'Halogen': [],
            'Hydrophobic_High': [],
            'Hydrophobic_Medium': [],
            'Hydrophilic': [],
            'Special_Cases': []
        }
        
        for idx, row in self.df.iterrows():
            # 根据logP分配到疏水性类别
            if pd.notna(row['LogP']):
                if row['LogP'] > 4:
                    categories['Hydrophobic_High'].append(idx)
                elif 2 < row['LogP'] <= 4:
                    categories['Hydrophobic_Medium'].append(idx)
                else:
                    categories['Hydrophilic'].append(idx)
            
            # 根据官能团分配到结构类别
            if row['is_aromatic']:
                categories['Aromatic'].append(idx)
            elif not row['is_aromatic']:
                categories['Aliphatic'].append(idx)
            
            if row['has_amine'] or row['has_amide'] or row['has_nitro']:
                categories['Nitrogen'].append(idx)
            
            if row['has_alcohol'] or row['has_carboxylic_acid'] or row['has_ketone'] or row['has_ester']:
                categories['Oxygen'].append(idx)
            
            if row['has_sulfur'] or row['has_phosphorus']:
                categories['Sulfur_Phosphorus'].append(idx)
            
            if row['has_halogen']:
                categories['Halogen'].append(idx)
        
        return categories
    
    def select_from_category(self, indices, n_select=4):
        """从指定类别的化合物中选择代表性化合物"""
        if len(indices) == 0:
            return []
        
        sub_df = self.df.loc[indices]
        
        # 按RTI值排序并均匀选择
        sub_df = sub_df.sort_values('RTI')
        selected = []
        
        # 如果数量足够，选择首、中、尾部的化合物
        if len(sub_df) >= n_select:
            step = max(1, len(sub_df) // n_select)
            for i in range(0, len(sub_df), step):
                if len(selected) >= n_select:
                    break
                selected.append(sub_df.index[i])
        else:
            selected = sub_df.index.tolist()
        
        return selected
    
    def run_selection(self, target_total=45):
        """运行完整的选择流程"""
        self.preprocess_data()
        categories = self.categorize_compounds()
        
        print("\n各类别化合物数量统计:")
        for cat, indices in categories.items():
            print(f"{cat}: {len(indices)}个")
        
        # 为每个类别分配选择数量（按比例）
        total_compounds = sum(len(indices) for indices in categories.values())
        selection_plan = {}
        
        for cat, indices in categories.items():
            if cat not in ['Hydrophobic_High', 'Hydrophobic_Medium', 'Hydrophilic']:
                # 结构类别：按比例分配
                proportion = len(indices) / total_compounds
                n_select = max(2, min(6, int(proportion * target_total)))
                selection_plan[cat] = n_select
        
        # 确保总数为目标值
        current_total = sum(selection_plan.values())
        if current_total < target_total:
            # 将剩余名额分配给化合物数量多的类别
            remaining = target_total - current_total
            sorted_cats = sorted(selection_plan.items(), key=lambda x: len(categories[x[0]]), reverse=True)
            for i in range(min(remaining, len(sorted_cats))):
                selection_plan[sorted_cats[i][0]] += 1
        
        # 执行选择
        all_selected = []
        for cat, n_select in selection_plan.items():
            selected = self.select_from_category(categories[cat], n_select)
            all_selected.extend(selected)
            print(f"{cat}: 选择了{len(selected)}个化合物")
        
        # 去重
        all_selected = list(set(all_selected))
        
        # 保存结果
        result_df = self.df.loc[all_selected]
        result_df.to_csv('selected_compounds.csv', index=False)
        
        print(f"\n选择完成！共选择了{len(all_selected)}个化合物")
        print("结果已保存到 selected_compounds.csv")
        
        return result_df

# 使用示例
if __name__ == "__main__":
    # 假设你的数据文件名为 'norman_data.csv'，包含 'SMILES' 和 'RTI' 列
    selector = CompoundSelector('norman_data.csv')
    selected_compounds = selector.run_selection(target_total=45)
```

---

## **四、手动检查与调整清单**

自动选择后，你需要手动检查这些化合物的**标准品可得性**。这里是一个检查清单：

| 类别 | 亚类 | 选择的化合物 | CAS号 | 是否易购 | 备注 |
|------|------|--------------|-------|----------|------|
| **芳香化合物** | 单环芳烃 | 苯 | 71-43-2 | ✓ | 基础化学品 |
| | 多环芳烃 | 萘 | 91-20-3 | ✓ | 易购 |
| | 芳香醇 | 苯酚 | 108-95-2 | ✓ | 易购 |
| | 芳香酸 | 苯甲酸 | 65-85-0 | ✓ | 易购 |
| **含氮化合物** | 伯胺 | 苯胺 | 62-53-3 | ✓ | 易购 |
| | 酰胺 | 乙酰胺 | 60-35-5 | ✓ | 易购 |
| | 硝基化合物 | 硝基苯 | 98-95-3 | ✓ | 易购 |
| **含氧化合物** | 伯醇 | 乙醇 | 64-17-5 | ✓ | 实验室常用 |
| | 羧酸 | 乙酸 | 64-19-7 | ✓ | 实验室常用 |
| | 酮 | 丙酮 | 67-64-1 | ✓ | 实验室常用 |
| **含卤化合物** | 氯代物 | 氯苯 | 108-90-7 | ✓ | 易购 |
| | 溴代物 | 溴苯 | 108-86-1 | ✓ | 易购 |
| **特殊案例** | PAH异构体 | 苯并[a]芘 | 50-32-8 | ✓ | 需特殊订购 |

---

## **五、最后的建议**

1. **先用自动脚本生成候选列表**（约50-60个化合物）
2. **批量查询这些化合物的标准品**（用CAS号批量查询Sigma、AccuStandard等供应商）
3. **对买不到或太贵的化合物**，在同一亚类内寻找替代品
4. **最终确定40-45个标尺化合物** + **2-3组验证案例化合物**

这个流程能确保你的标尺化合物：
1. **化学空间覆盖全面**（十大类别，每个类别内的亚类）
2. **保留时间范围覆盖**（从亲水到强疏水）
3. **实际可行性高**（标准品易得）

现在你可以开始运行这个脚本了。如果有任何技术问题（如RDKit安装、数据格式），我可以继续帮你解决。