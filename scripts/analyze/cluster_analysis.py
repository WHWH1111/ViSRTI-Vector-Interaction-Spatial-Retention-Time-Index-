import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter

from rdkit import Chem
from rdkit.Chem import Descriptors

def classify_compounds(df):
    """
    根据分子结构和物化性质对化合物进行分类（使用RDKit）
    返回分类结果和每个化合物的标签列表
    """
    classifications = {
        'Aromatic': [],           # 芳香化合物
        'Aliphatic': [],          # 脂肪族化合物  
        'Nitrogen_containing': [], # 含氮化合物（包括胺类、酰胺类、硝基化合物等）
        'Oxygen_containing': [],   # 含氧化合物（包括醇/酚类、羧酸类、羰基化合物等）
        'Sulfur_containing': [],  # 含硫化合物
        'Phosphorus_containing': [], # 含磷化合物
        'Halogen_containing': [], # 含卤化合物
        'High_hydrophobic': [],   # 高疏水性（logKow > 4）
        'Medium_hydrophobic': [], # 中等疏水性（2 < logKow <= 4）
        'Hydrophilic': []         # 亲水性（logKow <= 2）
    }
    
    # 为每个化合物存储标签列表
    compound_tags = {idx: [] for idx in df.index}
    
    for idx, row in df.iterrows():
        smiles = str(row['SMILES'])
        log_kow = row.get('logKow_EPISuite', np.nan)
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
                
            # 芳香性判断
            if any(atom.GetIsAromatic() for atom in mol.GetAtoms()):
                classifications['Aromatic'].append(idx)
                compound_tags[idx].append('Aromatic')
            else:
                classifications['Aliphatic'].append(idx)
                compound_tags[idx].append('Aliphatic')
            
            # 含氮官能团 - 归入含氮化合物大类
            if (mol.HasSubstructMatch(Chem.MolFromSmarts('[NX3;H2,H1;!$(NC=O)]')) or 
                mol.HasSubstructMatch(Chem.MolFromSmarts('C(=O)N')) or
                mol.HasSubstructMatch(Chem.MolFromSmarts('[N+](=O)[O-]'))):
                classifications['Nitrogen_containing'].append(idx)
                compound_tags[idx].append('Nitrogen_containing')
            
            # 含氧官能团 - 归入含氧化合物大类
            if (mol.HasSubstructMatch(Chem.MolFromSmarts('[OH]')) or
                mol.HasSubstructMatch(Chem.MolFromSmarts('C(=O)O')) or
                mol.HasSubstructMatch(Chem.MolFromSmarts('C=O'))):
                classifications['Oxygen_containing'].append(idx)
                compound_tags[idx].append('Oxygen_containing')
            
            # 硫/磷化合物
            if any(atom.GetSymbol() == 'S' for atom in mol.GetAtoms()):
                classifications['Sulfur_containing'].append(idx)
                compound_tags[idx].append('Sulfur_containing')
            if any(atom.GetSymbol() == 'P' for atom in mol.GetAtoms()):
                classifications['Phosphorus_containing'].append(idx)
                compound_tags[idx].append('Phosphorus_containing')
            
            # 卤素化合物
            halogens = ['F', 'Cl', 'Br', 'I']
            if any(atom.GetSymbol() in halogens for atom in mol.GetAtoms()):
                classifications['Halogen_containing'].append(idx)
                compound_tags[idx].append('Halogen_containing')
            
            # 疏水性分类
            if not np.isnan(log_kow):
                if log_kow > 4:
                    classifications['High_hydrophobic'].append(idx)
                    compound_tags[idx].append('High_hydrophobic')
                elif log_kow > 2:
                    classifications['Medium_hydrophobic'].append(idx)
                    compound_tags[idx].append('Medium_hydrophobic')
                else:
                    classifications['Hydrophilic'].append(idx)
                    compound_tags[idx].append('Hydrophilic')
                    
        except Exception as e:
            print(f"Error processing compound {idx}: {e}")
            continue
    
    return classifications, compound_tags

def save_compound_tags_to_csv(df, compound_tags, output_path):
    """
    将每个化合物的标签列表保存为CSV文件
    """
    # 创建包含SMILES和标签列表的DataFrame
    tags_data = []
    for idx, tags in compound_tags.items():
        if idx in df.index:
            smiles = df.loc[idx, 'SMILES']
            # 将标签列表转换为逗号分隔的字符串
            tags_str = ','.join(tags) if tags else 'Unclassified'
            tags_data.append({
                'SMILES': smiles,
                'Tags': tags_str,
            })
    
    tags_df = pd.DataFrame(tags_data)
    
    # 保存到CSV文件
    tags_df.to_csv(output_path, index=False)
    print(f"化合物标签已保存到: {output_path}")
    
    # 也保存一个更详细的版本，每个标签单独一列
    detailed_data = []
    all_tags = set()
    for tags in compound_tags.values():
        all_tags.update(tags)
    
    for idx, tags in compound_tags.items():
        if idx in df.index:
            row_data = {'SMILES': df.loc[idx, 'SMILES']}
            for tag in all_tags:
                row_data[tag] = 1 if tag in tags else 0
            detailed_data.append(row_data)
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_output_path = output_path.replace('.csv', '_detailed.csv')
    detailed_df.to_csv(detailed_output_path, index=False)
    print(f"详细标签矩阵已保存到: {detailed_output_path}")
    
    return tags_df, detailed_df

def calculate_performance_by_category(df, classifications, property_mean=None, property_std=None):
    """
    计算每个类别的预测性能
    """
    performance_data = []
    
    # 如果提供了均值和标准差，则使用它们进行反标准化
    if property_mean is not None and property_std is not None:
        use_inverse_normalization = True
    else:
        use_inverse_normalization = False
    
    for category, indices in classifications.items():
        if len(indices) < 30:  # 跳过数量少于30的类别
            continue
            
        subset = df.iloc[indices]
        predicted = subset['Predicted']
        actual = subset['Pred_RTI_Positive_ESI']
        
        # 确保没有NaN值
        mask = ~(np.isnan(predicted) | np.isnan(actual))
        predicted = predicted[mask]
        actual = actual[mask]
        
        if len(predicted) < 2:  # 至少需要2个点来计算指标
            continue
        
        # 如果需要，对数据进行标准化（因为输入数据已经是反标准化的）
        if use_inverse_normalization:
            predicted = (predicted - property_mean) / property_std
            actual = (actual - property_mean) / property_std
            
        # 计算性能指标
        mse = np.mean((predicted - actual) ** 2)
        mae = np.mean(np.abs(predicted - actual))
        rmse = np.sqrt(np.mean((predicted - actual) ** 2))
        
        # 计算R_squared
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        if ss_tot == 0:
            r_squared = 0.0
        else:
            r_squared = 1 - (ss_res / ss_tot)
        
        performance_data.append({
            'Category': category,
            'Count': len(indices),
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R_squared': r_squared
        })
    
    return pd.DataFrame(performance_data)

def plot_category_performance(performance_df):
    """
    绘制每个类别的预测性能
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # MSE
    axes[0, 0].bar(performance_df['Category'], performance_df['MSE'], color='lightcoral')
    axes[0, 0].set_title('Mean Squared Error by Category')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # MAE
    axes[0, 1].bar(performance_df['Category'], performance_df['MAE'], color='skyblue')
    axes[0, 1].set_title('Mean Absolute Error by Category')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # RMSE
    axes[0, 2].bar(performance_df['Category'], performance_df['RMSE'], color='lightgreen')
    axes[0, 2].set_title('Root Mean Square Error by Category')
    axes[0, 2].set_ylabel('RMSE')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # R_squared
    axes[1, 0].bar(performance_df['Category'], performance_df['R_squared'], color='gold')
    axes[1, 0].set_title('R-squared by Category')
    axes[1, 0].set_ylabel('R-squared')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 化合物数量
    axes[1, 1].bar(performance_df['Category'], performance_df['Count'], color='purple')
    axes[1, 1].set_title('Compound Count by Category')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 隐藏多余的子图
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('data/summary/category_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_comprehensive_analysis(performance_df, cross_category_df, category_counts):
    """
    绘制综合分析图，包含每个类别的化合物数量、跨类别情况以及预测性能
    """
    from matplotlib.colors import LinearSegmentedColormap
    
    # 准备数据
    categories = performance_df['Category'].tolist()
    counts = performance_df['Count'].tolist()
    
    # 创建图形和子图
    fig = plt.figure(figsize=(20, 12))
    
    # 第一个子图：每个类别的化合物数量（柱状图）
    ax1 = plt.subplot(2, 3, 1)
    bars = ax1.bar(range(len(categories)), counts, 
                   color='lightblue', edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Categories')
    ax1.set_ylabel('Number of Compounds')
    ax1.set_title('Compound Count by Category')
    ax1.set_xticks(range(len(categories)))
    ax1.set_xticklabels(categories, rotation=45, ha='right')
    
    # 在柱子上显示数值
    for i, (category, count) in enumerate(zip(categories, counts)):
        ax1.text(i, count + max(counts) * 0.01, str(count), 
                 ha='center', va='bottom', fontsize=8)
    
    # 第二个子图：跨类别情况（热图）
    ax2 = plt.subplot(2, 3, 2)
    if not cross_category_df.empty:
        cross_pivot = cross_category_df.pivot(index='Category1', columns='Category2', values='Count').fillna(0)
        # 确保行列顺序一致
        cross_pivot = cross_pivot.reindex(index=categories, columns=categories, fill_value=0)
        im = ax2.imshow(cross_pivot.values, cmap='Blues', aspect='auto')
        
        # 设置刻度标签
        ax2.set_xticks(range(len(categories)))
        ax2.set_yticks(range(len(categories)))
        ax2.set_xticklabels(categories, rotation=45, ha='right', fontsize=8)
        ax2.set_yticklabels(categories, fontsize=8)
        ax2.set_title('Cross-category Membership')
        
        # 添加数值标注
        for i in range(len(categories)):
            for j in range(len(categories)):
                if cross_pivot.iloc[i, j] > 0:
                    ax2.text(j, i, f'{int(cross_pivot.iloc[i, j])}', 
                             ha='center', va='center', fontsize=7)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
        cbar.set_label('Count')
    else:
        ax2.text(0.5, 0.5, 'No cross-category data', ha='center', va='center', 
                 transform=ax2.transAxes)
        ax2.set_title('Cross-category Membership')
    
    # 第三个子图：预测性能（用颜色深浅表示MSE性能）
    ax3 = plt.subplot(2, 3, 3)
    # 使用MSE值（越小越好）来确定颜色深浅，需要反转数值以便直观理解
    mse_values = performance_df['MSE'].tolist()
    # 归一化到0-1范围，用于颜色映射
    normalized_mse = [(mse - min(mse_values) + 0.01) / (max(mse_values) - min(mse_values) + 0.02) 
                      for mse in mse_values]
    
    # 定义红蓝渐变（红色表示较好性能）
    colors = [(1, 0, 0), (0, 0, 1)]  # 红色到蓝色
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
    bar_colors = [cmap(norm_mse) for norm_mse in normalized_mse]
    
    bars = ax3.bar(range(len(categories)), mse_values, color=bar_colors, edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('Categories')
    ax3.set_ylabel('MSE')
    ax3.set_title('Prediction Performance by Category (MSE)')
    ax3.set_xticks(range(len(categories)))
    ax3.set_xticklabels(categories, rotation=45, ha='right')
    
    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, 
                               norm=plt.Normalize(vmin=min(mse_values), vmax=max(mse_values)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax3, shrink=0.8)
    cbar.set_label('MSE (Lower is Better - Red is Better)')
    
    # 第四个子图：预测性能（用颜色深浅表示MAE性能）
    ax4 = plt.subplot(2, 3, 4)
    # 使用MAE值（越小越好）来确定颜色深浅，需要反转数值以便直观理解
    mae_values = performance_df['MAE'].tolist()
    # 归一化到0-1范围，用于颜色映射
    normalized_mae = [(mae - min(mae_values) + 0.01) / (max(mae_values) - min(mae_values) + 0.02) 
                      for mae in mae_values]
    
    # 定义红蓝渐变（红色表示较好性能）
    colors = [(1, 0, 0), (0, 0, 1)]  # 红色到蓝色
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
    bar_colors = [cmap(norm_mae) for norm_mae in normalized_mae]
    
    bars = ax4.bar(range(len(categories)), mae_values, color=bar_colors, edgecolor='black', linewidth=0.5)
    ax4.set_xlabel('Categories')
    ax4.set_ylabel('MAE')
    ax4.set_title('Prediction Performance by Category (MAE)')
    ax4.set_xticks(range(len(categories)))
    ax4.set_xticklabels(categories, rotation=45, ha='right')
    
    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, 
                               norm=plt.Normalize(vmin=min(mae_values), vmax=max(mae_values)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax4, shrink=0.8)
    cbar.set_label('MAE (Lower is Better - Red is Better)')
    
    # 第五个子图：R²值（越高越好）
    ax5 = plt.subplot(2, 3, 5)
    r2_values = performance_df['R_squared'].tolist()
    # 归一化到0-1范围
    normalized_r2 = [(r2 - min(r2_values) + 0.01) / (max(r2_values) - min(r2_values) + 0.02) 
                     for r2 in r2_values]
    
    # 定义红蓝渐变（红色表示较好性能）
    colors = [(1, 0, 0), (0, 0, 1)]  # 红色到蓝色
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
    bar_colors = [cmap(norm_r2) for norm_r2 in normalized_r2]
    
    bars = ax5.bar(range(len(categories)), r2_values, color=bar_colors, edgecolor='black', linewidth=0.5)
    ax5.set_xlabel('Categories')
    ax5.set_ylabel('R²')
    ax5.set_title('Prediction Performance by Category (R²)')
    ax5.set_xticks(range(len(categories)))
    ax5.set_xticklabels(categories, rotation=45, ha='right')
    
    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, 
                               norm=plt.Normalize(vmin=min(r2_values), vmax=max(r2_values)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax5, shrink=0.8)
    cbar.set_label('R² (Higher is Better - Red is Better)')
    
    # 隐藏多余的子图
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('data/summary/comprehensive_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def analyze_cross_category_membership(df, classifications):
    """
    分析化合物的跨类别归属情况
    """
    # 创建一个矩阵来表示化合物属于哪些类别
    compound_category_matrix = pd.DataFrame(
        index=df.index,
        columns=list(classifications.keys())
    ).fillna(0)
    
    for category, indices in classifications.items():
        compound_category_matrix.loc[indices, category] = 1
    
    # 统计每个化合物属于的类别数
    category_counts = compound_category_matrix.sum(axis=1)
    
    # 统计各类别的交叉情况
    cross_category_data = []
    for category1 in classifications.keys():
        for category2 in classifications.keys():
            count = ((compound_category_matrix[category1] == 1) & 
                    (compound_category_matrix[category2] == 1)).sum()
            if count > 0:
                cross_category_data.append({
                    'Category1': category1,
                    'Category2': category2,
                    'Count': count
                })
    
    return pd.DataFrame(cross_category_data), category_counts

def plot_cross_category_analysis(cross_category_df, category_counts):
    """
    绘制跨类别分析图
    """
    # 绘制交叉类别热图
    if not cross_category_df.empty:
        cross_pivot = cross_category_df.pivot(index='Category1', columns='Category2', values='Count').fillna(0)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cross_pivot, annot=True, fmt='.0f', cmap='Blues')
        plt.title('Cross-category Membership')
        plt.tight_layout()
        plt.savefig('data/summary/cross_category_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 绘制化合物类别数分布
    plt.figure(figsize=(10, 6))
    unique_counts = category_counts.value_counts().sort_index()
    bars = plt.bar(unique_counts.index, unique_counts.values, color='purple', alpha=0.7, edgecolor='black', linewidth=0.5)
    for i, (x, y) in enumerate(zip(unique_counts.index, unique_counts.values)):
        plt.text(x, y + 0.5, str(y), ha='center', va='bottom', fontsize=9)
    plt.xlabel('Number of Categories per Compound')
    plt.ylabel('Number of Compounds')
    plt.title('Distribution of Category Membership')
    plt.tight_layout()
    plt.savefig('data/summary/category_membership_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
TASK_NAME = 'neg-3-mask(koc)-train_20251128-162233_dim48_layerH6_layerO6_batch64_lr0.0001_iter150'

def main():
    # 读取数据
    base_path = f"./log/visnet-v2/{TASK_NAME}"
    
    # 读取三个数据集
    dev_df = pd.read_csv(f"{base_path}/train_prediction_dev.txt", sep='\t')
    test_df = pd.read_csv(f"{base_path}/train_prediction_test.txt", sep='\t')
    train_df = pd.read_csv(f"{base_path}/train_prediction_train.txt", sep='\t')
    
    # 合并数据集
    df = pd.concat([dev_df, test_df, train_df], ignore_index=True)
    
    # 重命名列以匹配后续代码的期望
    df.rename(columns={'Smiles': 'SMILES', 'Correct': 'Pred_RTI_Positive_ESI', 'Predict': 'Predicted'}, inplace=True)
    
    # 读取训练参数获取均值和标准差
    import json
    training_params_path = f"./log/visnet-v2/{TASK_NAME}/training_params.json"
    try:
        with open(training_params_path, 'r') as f:
            training_params = json.load(f)
        property_mean = training_params["property_mean"]
        property_std = training_params["property_std"]
        print(f"使用训练参数中的均值和标准差: mean={property_mean}, std={property_std}")
    except FileNotFoundError:
        print("未找到训练参数文件，将使用原始数据计算指标")
        property_mean = None
        property_std = None
    except KeyError as e:
        print(f"训练参数文件中缺少必要的键: {e}，将使用原始数据计算指标")
        property_mean = None
        property_std = None
    
    print("数据加载完成，共有 {} 个化合物".format(len(df)))
    
    # 化合物分类
    print("正在进行化合物分类...")
    classifications, compound_tags = classify_compounds(df)
    
    # 保存化合物标签到CSV文件
    print("保存化合物标签...")
    tags_output_path = 'data/summary/compound_tags.csv'
    tags_df, detailed_df = save_compound_tags_to_csv(df, compound_tags, tags_output_path)
    
    # 过滤掉数量少于30的类别
    filtered_classifications = {k: v for k, v in classifications.items() if len(v) >= 30}
    print("保留的类别：", list(filtered_classifications.keys()))
    
    # 计算各类别的预测性能
    print("计算预测性能...")
    performance_df = calculate_performance_by_category(df, filtered_classifications, property_mean, property_std)
    print(performance_df)
    
    # 保存性能数据
    performance_df.to_csv('data/summary/category_performance.csv', index=False)
    
    # 绘制性能图表
    print("绘制性能图表...")
    plot_category_performance(performance_df)
    
    # 分析跨类别归属
    print("分析跨类别归属...")
    cross_category_df, category_counts = analyze_cross_category_membership(df, filtered_classifications)
    
    # 绘制跨类别分析图
    print("绘制跨类别分析图...")
    plot_cross_category_analysis(cross_category_df, category_counts)
    
    # 绘制综合分析图
    print("绘制综合分析图...")
    plot_comprehensive_analysis(performance_df, cross_category_df, category_counts)
    
    # 输出统计信息
    print("\n=== 分析结果 ===")
    print("1. 各类别化合物数量:")
    for category, indices in filtered_classifications.items():
        print(f"   {category}: {len(indices)} compounds")
    
    print("\n2. 性能最好的类别 (按MAE排序):")
    if not performance_df.empty:
        sorted_performance = performance_df.sort_values('MAE')
        for _, row in sorted_performance.iterrows():
            print(f"   {row['Category']}: MSE={row['MSE']:.2f}, MAE={row['MAE']:.2f}, RMSE={row['RMSE']:.2f}, R²={row['R_squared']:.2f}")
    else:
        print("   没有足够的数据计算性能指标")
    
    print("\n3. 跨类别统计:")
    print(f"   平均每个化合物属于 {category_counts.mean():.1f} 个类别")
    print(f"   最多一个化合物属于 {category_counts.max()} 个类别")
    
    print("\n4. 标签文件已生成:")
    print(f"   - 基础标签文件: {tags_output_path}")
    print(f"   - 详细标签矩阵: {tags_output_path.replace('.csv', '_detailed.csv')}")

if __name__ == "__main__":
    main()