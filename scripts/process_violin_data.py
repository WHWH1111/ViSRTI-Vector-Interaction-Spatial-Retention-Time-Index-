import json
import csv

target_name = "neg-violin_20251129-163119"

### —————————— ###

# 读取 JSON 文件
json_file_path = rf'd:\Projects\python\gnn-rt-1\results\violin\{target_name}\violin_plot_data.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)

# 提取 R² 和 MAE 数据
gnn_rt_r2 = data['r2_values']['GNN-RT']
visnet_v2_r2 = data['r2_values']['VisNet-V2']
gnn_rt_mae = data['mae_values']['GNN-RT']
visnet_v2_mae = data['mae_values']['VisNet-V2']

# 确保所有列表长度一致
assert len(gnn_rt_r2) == len(visnet_v2_r2) == len(gnn_rt_mae) == len(visnet_v2_mae), "数据长度不匹配"

# 写入 CSV 文件
csv_file_path = rf'd:\Projects\python\gnn-rt-1\results\violin\{target_name}\violin_plot_data.csv'
with open(csv_file_path, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # 写入表头
    writer.writerow(['GNN-RT_r2', 'VisNet-V2_r2', 'GNN-RT_mae', 'VisNet-V2_mae'])
    # 按行写入每个数据点
    for i in range(len(gnn_rt_r2)):
        writer.writerow([
            gnn_rt_r2[i],
            visnet_v2_r2[i],
            gnn_rt_mae[i],
            visnet_v2_mae[i]
        ])

print(f"CSV文件已成功生成: {csv_file_path}")