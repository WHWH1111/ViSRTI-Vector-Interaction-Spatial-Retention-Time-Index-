#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从enhanced_dataset.csv中筛选具有代表性的化合物
确保覆盖10大类及其亚分类，同时考虑RTI范围和分子量
"""

from typing import final
import pandas as pd
import numpy as np
from collections import defaultdict
import random
import os


def parse_tags(tag_string):
    """
    解析标签字符串，正确处理包含括号的内容
    例如："Aromatic(Polyaromatic,Heterocyclic),Halogen_containing(Chlorine),Hydrophilic"
    应该被解析为 ["Aromatic(Polyaromatic,Heterocyclic)", "Halogen_containing(Chlorine)", "Hydrophilic"]
    对于新格式："Nitrogen(Secondary_Amine,Tertiary_Amine,Amide);Oxygen(Ketone)"
    应该被解析为 ["Nitrogen(Secondary_Amine,Tertiary_Amine,Amide)", "Oxygen(Ketone)"]
    """
    if pd.isna(tag_string):
        return []
    
    # 首先尝试按分号分割（新格式）
    if ';' in tag_string:
        # 新格式：分号分隔主要类别
        parts = tag_string.split(';')
        tags = []
        for part in parts:
            part = part.strip()
            if part:
                tags.append(part)
        return tags
    
    # 旧格式：逗号分隔，需要处理括号
    tags = []
    i = 0
    while i < len(tag_string):
        if tag_string[i] == ',':
            i += 1
            continue
            
        # 查找标签开始位置
        start = i
        if tag_string[i] != '(':
            # 处理普通标签或带括号的标签（如 Aromatic(...)）
            j = i
            paren_count = 0
            while j < len(tag_string):
                if tag_string[j] == '(':
                    paren_count += 1
                elif tag_string[j] == ')':
                    paren_count -= 1
                elif tag_string[j] == ',' and paren_count == 0:
                    break
                j += 1
            tags.append(tag_string[start:j].strip())
            i = j + 1 if j < len(tag_string) else j
        else:
            # 处理以括号开头的标签（这种情况应该不会出现，但为了健壮性考虑）
            # 找到匹配的右括号
            paren_count = 1
            j = i + 1
            while j < len(tag_string) and paren_count > 0:
                if tag_string[j] == '(':
                    paren_count += 1
                elif tag_string[j] == ')':
                    paren_count -= 1
                j += 1
            # 现在j指向')'之后的字符
            # 找到下一个逗号或字符串结尾
            k = j
            while k < len(tag_string) and tag_string[k] != ',':
                k += 1
            tags.append(tag_string[start:k].strip())
            i = k + 1 if k < len(tag_string) else k
    
    return tags


def categorize_compounds(df):
    """
    根据Enhanced_Tags列对化合物进行分类
    """
    # 主要类别
    major_categories = [
        'Aromatic', 'Aliphatic', 'Nitrogen_containing', 'Oxygen_containing',
        'Sulfur_containing', 'Phosphorus_containing', 'Halogen_containing',
        'High_hydrophobic', 'Medium_hydrophobic', 'Hydrophilic'
    ]
    
    # 初始化分类字典
    categorized = {cat: [] for cat in major_categories}
    
    for idx, row in df.iterrows():
        tags = row['category_tag']  # 修改为使用新的列名
        if pd.isna(tags):
            continue
            
        # 使用新的解析方法
        tag_list = parse_tags(tags)
        
        # 按主要类别分类
        for category in major_categories:
            # 检查是否有完全匹配的主要类别或者以该类别开头的标签
            for tag in tag_list:
                # 直接匹配主要类别
                if tag == category:
                    categorized[category].append(idx)
                    break
                # 匹配以主要类别开头的标签（处理"Aromatic(...)"这样的情况）
                elif tag.startswith(category + '('):
                    categorized[category].append(idx)
                    break
                
    return categorized


def get_subcategories(tag_string, category=None):
    """
    从标签字符串中提取亚分类
    如果提供了category参数，则只提取该类别的亚分类
    """
    if pd.isna(tag_string):
        return []
    
    # 使用新的解析方法
    tag_list = parse_tags(tag_string)
        
    subcats = []
    
    # 如果指定了category，则只提取该category的亚分类
    if category:
        for tag in tag_list:
            if tag.startswith(category + '(') and '(' in tag and ')' in tag:
                # 提取括号内的内容作为亚分类
                start = tag.find('(')
                end = tag.find(')')
                if start != -1 and end != -1:
                    subcat = tag[start+1:end]
                    # 处理多个亚分类的情况，如"Polyaromatic,Heterocyclic,PAH,Polycyclic_Aromatic"
                    # 注意这里也要处理分号和逗号的不同情况
                    if ';' in subcat:
                        # 如果子类别内部也有分号分隔
                        subcats.extend([s.strip() for s in subcat.split(';')])
                    else:
                        # 逗号分隔的子类别
                        subcats.extend([s.strip() for s in subcat.split(',')])
        return subcats
    
    # 如果没有指定category，则提取所有亚分类
    for tag in tag_list:
        if '(' in tag and ')' in tag:
            # 提取括号内的内容作为亚分类
            start = tag.find('(')
            end = tag.find(')')
            if start != -1 and end != -1:
                subcat = tag[start+1:end]
                # 处理多个亚分类的情况
                if ';' in subcat:
                    # 如果子类别内部也有分号分隔
                    subcats.extend([s.strip() for s in subcat.split(';')])
                else:
                    # 逗号分隔的子类别
                    subcats.extend([s.strip() for s in subcat.split(',')])
                
    return subcats


def safe_qcut(series, q, labels):
    """
    安全的分位数切割，处理数据不足的情况
    """
    if len(series) < q:
        # 如果数据点不够分组，就用简单的分割
        return pd.Series([labels[min(i, len(labels)-1)] for i in range(len(series))], index=series.index)
    
    try:
        return pd.qcut(series, q=q, labels=labels, duplicates='drop')
    except ValueError:
        # 尝试减少分组数量
        return pd.qcut(series, q=min(len(labels), len(set(series))), labels=labels[:min(len(labels), len(set(series)))], duplicates='drop')


def select_representatives(df, categorized, target_count=70):
    """
    从每个主要类别中选择具有代表性的化合物
    
    选择标准：
    1. Pos-RTI 和 Neg-RTI 都存在
    2. 覆盖不同的RTI范围（低、中、高）
    3. 覆盖不同的分子量范围
    4. 覆盖不同的亚分类
    5. 每个类别选择适量的化合物
    """
    selected_indices = []
    
    # 为每个主要类别设定选择数量
    # 每个类别选择7个化合物
    category_targets = {
        'Aromatic': 7,
        'Aliphatic': 7,
        'Nitrogen_containing': 7,
        'Oxygen_containing': 7,
        'Sulfur_containing': 7,
        'Phosphorus_containing': 7,
        'Halogen_containing': 7,
        'High_hydrophobic': 7,
        'Medium_hydrophobic': 7,
        'Hydrophilic': 7
    }
    
    total_selected = 0
    
    # 重置索引以确保索引连续
    df_reset = df.reset_index(drop=False)
    # 创建原始索引到新索引的映射
    original_to_new_index = {original_idx: new_idx for new_idx, original_idx in enumerate(df.index)}
    
    # 更新categorized中的索引
    categorized_updated = {}
    for category, indices in categorized.items():
        categorized_updated[category] = [original_to_new_index[idx] for idx in indices if idx in original_to_new_index]
    
    selected_data_from_category = []  # 存储选中的实际数据
    globally_selected_original_indices = set()  # 跟踪全局已选择的原始索引，避免不同大类间重复选择
    
    for category, indices in categorized_updated.items():
        if total_selected >= target_count:
            break
            
        print(f"Processing category: {category} with {len(indices)} compounds")
        
        # 过滤掉已经被其他类别选择的化合物（基于原始索引）
        filtered_indices = [idx for idx in indices if df_reset.iloc[idx]['index'] not in globally_selected_original_indices]
        print(f"  After filtering out already selected compounds: {len(filtered_indices)} compounds")
        
        if not filtered_indices:
            print(f"  No compounds left for category {category} after filtering")
            continue
            
        # 获取该类别下的所有化合物数据
        category_data = []
        for idx in filtered_indices:
            row = df_reset.iloc[idx]
            rti_pos = row['Pos_Predicted']
            rti_neg = row['Neg_Predicted']
            mass_pos = row['Monoiso_Mass_pos'] if not pd.isna(row['Monoiso_Mass_pos']) else None
            mass_neg = row['Monoiso_Mass_neg'] if not pd.isna(row['Monoiso_Mass_neg']) else None
            
            # 确保分子量也存在
            if (mass_pos is not None or mass_neg is not None):
                mass = mass_pos if mass_pos is not None else mass_neg
                category_data.append({
                    'index': idx,
                    'original_index': df.index[idx],  # 保存原始索引
                    'rti_pos': rti_pos,
                    'rti_neg': rti_neg,
                    'mass': mass,
                    'category': category,
                    'subcategories': get_subcategories(row['category_tag'], category),  # 修改为使用新的列名
                    'row': row,
                })
        
        if len(category_data) == 0:
            continue
            
        # 确定该类别要选择的数量
        target_num = min(category_targets.get(category, 7), len(category_data), target_count - total_selected)
        
        # 转换为DataFrame以便处理
        category_df = pd.DataFrame(category_data)
        
        # 添加大类标记
        category_df['major_category'] = category
        
        # 按RTI分组（低、中、高）
        if len(category_df) >= 3:
            category_df['RTI_Pos_Group'] = safe_qcut(category_df['rti_pos'], q=3, labels=['Low', 'Medium', 'High'])
            category_df['RTI_Neg_Group'] = safe_qcut(category_df['rti_neg'], q=3, labels=['Low', 'Medium', 'High'])
        else:
            category_df['RTI_Pos_Group'] = 'Medium'  # 默认分组
            category_df['RTI_Neg_Group'] = 'Medium'  # 默认分组
            
        # 按分子量分组（低、中、高）
        if len(category_df) >= 3:
            category_df['Mass_Group'] = safe_qcut(category_df['mass'], q=3, labels=['Low', 'Medium', 'High'])
        else:
            category_df['Mass_Group'] = 'Medium'  # 默认分组
        
        # 为每个化合物添加亚分类数量
        category_df['Subcategory_Count'] = category_df['subcategories'].apply(len)
        
        # 优先选择具有更多亚分类的化合物
        selected_from_category = []
        
        # 先确保RTI和分子量的覆盖
        pos_rti_groups = category_df['RTI_Pos_Group'].unique()
        neg_rti_groups = category_df['RTI_Neg_Group'].unique()
        mass_groups = category_df['Mass_Group'].unique()
        
        # 创建一个组合分组，综合考虑所有因素
        category_df['combined_group'] = category_df['RTI_Pos_Group'].astype(str) + '_' + \
                                      category_df['RTI_Neg_Group'].astype(str) + '_' + \
                                      category_df['Mass_Group'].astype(str)
        
        # 按组合分组选择化合物
        combined_groups = category_df['combined_group'].unique()
        
        # 每个组合至少选择一个化合物（如果可能的话）
        for group in combined_groups:
            if len(selected_from_category) >= target_num:
                break
                
            group_df = category_df[category_df['combined_group'] == group]
            if len(group_df) == 0:
                continue
                
            # 在组内按亚分类数量排序，优先选择亚分类多的
            group_df = group_df.sort_values('Subcategory_Count', ascending=False)
            
            # 选择该组中最好的候选化合物
            selected_candidate = group_df.iloc[0]
            idx = selected_candidate['index']
            original_idx = selected_candidate['original_index']
            
            # 确保不重复选择
            if idx not in selected_from_category and original_idx not in globally_selected_original_indices:
                selected_from_category.append(idx)
                selected_data_from_category.append(selected_candidate)  # 添加实际数据
                globally_selected_original_indices.add(original_idx)  # 添加到全局已选择集合
        
        # 如果还需要更多化合物，按亚分类数量补充选择
        if len(selected_from_category) < target_num:
            remaining = category_df[~category_df['index'].isin(selected_from_category)]
            remaining = remaining.sort_values('Subcategory_Count', ascending=False)
            additional_needed = target_num - len(selected_from_category)
            additional_selection = remaining.head(additional_needed)
            for _, candidate in additional_selection.iterrows():
                idx = candidate['index']
                original_idx = candidate['original_index']
                # 确保不重复选择
                if idx not in selected_from_category and original_idx not in globally_selected_original_indices:
                    selected_from_category.append(idx)
                    selected_data_from_category.append(candidate)  # 添加实际数据
                    globally_selected_original_indices.add(original_idx)  # 添加到全局已选择集合
        
        selected_indices.extend(selected_from_category[:target_num])
        total_selected += len(selected_from_category[:target_num])
        print(f"Selected {len(selected_from_category[:target_num])} compounds from {category}")
    
    return selected_data_from_category


def main():
    # STAGE 读取数据
    file_path = 'predictions/visnet-v2-5fold/categorized_dataset.csv'
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    print("Reading dataset...")
    df = pd.read_csv(file_path, encoding='utf-8')
    print(f"Total compounds: {len(df)}")
    
    # 过滤出同时具有Pos和Neg RTI值的化合物
    df_filtered = df.dropna(subset=['Pos_Predicted', 'Neg_Predicted'])
    print(f"Compounds with both Pos and Neg RTI values: {len(df_filtered)}")
    
    # 添加过滤条件：要求Pos预测值和实际值相差不超过30%，Neg同理
    # 首先确保Actual值存在
    df_filtered = df_filtered.dropna(subset=['Pos_Actual', 'Neg_Actual'])
    
    # 计算相对误差并过滤
    pos_relative_error = abs(df_filtered['Pos_Predicted'] - df_filtered['Pos_Actual']) / df_filtered['Pos_Actual']
    neg_relative_error = abs(df_filtered['Neg_Predicted'] - df_filtered['Neg_Actual']) / df_filtered['Neg_Actual']
    
    # 过滤相对误差不超过30%的化合物
    df_filtered = df_filtered[(pos_relative_error <= 0.3) & (neg_relative_error <= 0.3)]
    print(f"Compounds with relative error <= 30% for both Pos and Neg: {len(df_filtered)}")
    
    # STAGE 分类化合物
    print("Categorizing compounds...")
    categorized = categorize_compounds(df_filtered)
    # 显示各类别统计
    print("\nCategory Statistics:")
    for category, indices in categorized.items():
        print(f"{category}: {len(indices)} compounds")
    
    # STAGE 选择代表性化合物
    print("\nSelecting representative compounds...")
    selected_data_from_category = select_representatives(df_filtered, categorized, target_count=70)
    print(f"\nTotal selected compounds: {len(selected_data_from_category)}")
    # print(selected_data_from_category[0].keys(), selected_data_from_category[0]['row'].keys())
    
    # 将 selected_data_from_category 转换为 DataFrame 并保存
    if selected_data_from_category:
        # 从两层结构中提取数据构建新的DataFrame
        # 外层字段：major_category, subcategories
        # 内层字段：Norman_SusDat_ID, Name_neg, Pos_Predicted, Neg_Predicted, Monoiso_Mass_neg
        
        output_data = []
        for item in selected_data_from_category:
            # 从外层提取字段
            major_category = item['major_category']
            subcategories = item['subcategories']
            
            # 从内层row提取字段
            row = item['row']
            smiles = row['SMILES']
            norman_id = row['Norman_SusDat_ID']
            name_neg = row['Name_neg']
            pos_rti = row['Pos_Predicted']
            neg_rti = row['Neg_Predicted']
            monoiso_mass = row['Monoiso_Mass_neg']
            all_tags = row['category_tag']  # 修改为使用新的列名
            
            output_data.append({
                'Norman_SusDat_ID': norman_id,
                'Name_neg': name_neg,
                "SMILES": smiles,
                'pos-RTI': pos_rti,
                'neg-RTI': neg_rti,
                'Monoiso_Mass': monoiso_mass,
                # 'mass': item['mass'],
                'major_category': major_category,
                'subcategories': subcategories,
                # 'all_tags': all_tags
            })
            
        output_df = pd.DataFrame(output_data)
        
        # 保存结果
        output_file = 'representative_compounds.csv'
        output_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\nResults saved to {output_file}")
    else:
        print("No compounds selected.")

if __name__ == "__main__":
    main()