import pandas as pd

# 读取原始数据
original_data_path = r'data/MMF-3/data1028_sheet0.csv'
original_df = pd.read_csv(original_data_path, low_memory=False)

# 读取负离子预测结果
neg_predictions_path = r'predictions\prediction_20251128-235305\None_data1028_sheet0_predictions.csv'
neg_df = pd.read_csv(neg_predictions_path)

# 读取正离子预测结果
pos_predictions_path = r'predictions/savepoint-1/pos-prediction_20251122-113848/None_data1028_sheet0_predictions.csv'
pos_df = pd.read_csv(pos_predictions_path)

# 重命名预测列以区分
neg_df.rename(columns={'Predicted': 'Predicted_Negative', 'Actual': 'Actual_Negative'}, inplace=True)
pos_df.rename(columns={'Predicted': 'Predicted_Positive', 'Actual': 'Actual_Positive'}, inplace=True)

# 合并数据
# 首先将原始数据与负离子预测结果合并
merged_df = pd.merge(original_df, neg_df[['Norman_SusDat_ID', 'Predicted_Negative', 'Actual_Negative']], 
                     on='Norman_SusDat_ID', how='left')

# 然后将正离子预测结果合并
merged_df = pd.merge(merged_df, pos_df[['Norman_SusDat_ID', 'Predicted_Positive', 'Actual_Positive']], 
                     on='Norman_SusDat_ID', how='left')

# 保存合并后的数据
output_path = r'scripts/data-merge/data1028_sheet0_with_predictions.csv'
merged_df.to_csv(output_path, index=False)

print(f"合并完成，结果已保存到: {output_path}")
print(f"原始数据行数: {len(original_df)}")
print(f"负离子预测行数: {len(neg_df)}")
print(f"正离子预测行数: {len(pos_df)}")
print(f"合并后数据行数: {len(merged_df)}")