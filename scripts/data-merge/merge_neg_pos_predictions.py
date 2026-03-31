import pandas as pd
import os

def merge_neg_pos_predictions():
    """
    合并neg和pos离子的预测结果到一个CSV文件中，按ID合并
    """
    # 定义文件路径
    neg_file = r'd:\Projects\python\gnn-rt-1\predictions\visnet-v2-5fold\neg\merged_predictions_with_tags.csv'
    pos_file = r'd:\Projects\python\gnn-rt-1\predictions\visnet-v2-5fold\pos\merged_predictions_with_tags.csv'
    output_dir = r'd:\Projects\python\gnn-rt-1\predictions\visnet-v2-5fold'
    
    # 读取neg和pos预测结果
    print("正在读取负离子预测结果...")
    df_neg = pd.read_csv(neg_file)
    
    print("正在读取正离子预测结果...")
    df_pos = pd.read_csv(pos_file)
    
    # 重命名列以区分正负离子数据
    df_neg.rename(columns={
        'Predicted': 'Neg_Predicted',
        'Actual': 'Neg_Actual'
    }, inplace=True)
    
    df_pos.rename(columns={
        'Predicted': 'Pos_Predicted',
        'Actual': 'Pos_Actual'
    }, inplace=True)
    
    # 合并数据，按ID合并
    print("正在合并数据...")
    merged_df = pd.merge(df_pos, df_neg, on=['Norman_SusDat_ID', 'SMILES', 'Tags'], how='outer')
    
    # 重新排序列
    columns_order = ['Norman_SusDat_ID', 'SMILES', 'Pos_Predicted', 'Pos_Actual', 'Neg_Predicted', 'Neg_Actual', 'Tags']
    merged_df = merged_df[columns_order]
    
    # 保存合并后的数据
    output_file = os.path.join(output_dir, 'merged_neg_pos_predictions_by_id.csv')
    merged_df.to_csv(output_file, index=False)
    print(f"合并完成！结果已保存到: {output_file}")
    print(f"总记录数: {len(merged_df)}")
    print(f"只有正离子数据的记录数: {merged_df['Pos_Predicted'].notna().sum()}")
    print(f"只有负离子数据的记录数: {merged_df['Neg_Predicted'].notna().sum()}")
    print(f"同时有正负离子数据的记录数: {merged_df[['Pos_Predicted', 'Neg_Predicted']].notna().all(axis=1).sum()}")

if __name__ == "__main__":
    merge_neg_pos_predictions()