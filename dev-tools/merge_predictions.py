import pandas as pd
import os

def merge_predictions():
    # 文件路径
    original_file = '/home/data2/rhj/project/gnn/gnn-1/data/MMF-2/MMF-GNN-valid-smiles.csv'
    neg_pred_file = '/home/data2/rhj/project/gnn/gnn-1/predictions/neg-prediction_20251028-012019/MMF-GNN-valid-smiles_simple_predictions.csv'
    pos_pred_file = '/home/data2/rhj/project/gnn/gnn-1/predictions/pos-prediction_20251028-012053/MMF-GNN-valid-smiles_simple_predictions.csv'
    output_file = '/home/data2/rhj/project/gnn/gnn-1/data/MMF-2/MMF-GNN_uncovered_with_predictions.csv'
    
    # 读取原始数据
    original_df = pd.read_csv(original_file)
    
    # 读取负离子预测数据
    neg_pred_df = pd.read_csv(neg_pred_file)
    neg_pred_dict = dict(zip(neg_pred_df['SMILES'], neg_pred_df['Predicted']))
    
    # 读取正离子预测数据
    pos_pred_df = pd.read_csv(pos_pred_file)
    pos_pred_dict = dict(zip(pos_pred_df['SMILES'], pos_pred_df['Predicted']))
    
    # 在指定列后插入新的预测列
    # 在'Uncertainty_RTI_pos'后插入'Predicted_RTI_Positive_ESI_New'
    pos_col_index = original_df.columns.get_loc('Uncertainty_RTI_pos') + 1
    original_df.insert(pos_col_index, 'Predicted_RTI_Positive_ESI_New', None)
    
    # 在'Uncertainty_RTI_neg'后插入'Predicted_RTI_Negative_ESI_New'
    neg_col_index = original_df.columns.get_loc('Uncertainty_RTI_neg') + 1
    original_df.insert(neg_col_index, 'Predicted_RTI_Negative_ESI_New', None)
    
    # 根据SMILES匹配填充预测值
    for index, row in original_df.iterrows():
        smiles = row['SMILES']
        
        # 查找负离子预测值
        if smiles in neg_pred_dict:
            original_df.at[index, 'Predicted_RTI_Negative_ESI_New'] = neg_pred_dict[smiles]
            
        # 查找正离子预测值
        if smiles in pos_pred_dict:
            original_df.at[index, 'Predicted_RTI_Positive_ESI_New'] = pos_pred_dict[smiles]
    
    # 保存到新文件
    original_df.to_csv(output_file, index=False)
    print(f"Merged dataset saved to {output_file}")
    print(f"Original dataset shape: {original_df.shape}")
    print(f"Negative predictions matched: {original_df['Predicted_RTI_Negative_ESI_New'].notna().sum()}")
    print(f"Positive predictions matched: {original_df['Predicted_RTI_Positive_ESI_New'].notna().sum()}")

if __name__ == "__main__":
    merge_predictions()