import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator

# 设置英文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建示例数据
np.random.seed(42)
categories = ['Feature A', 'Feature B', 'Feature C', 'Feature D', 'Feature E']

# 为蜂群图创建数据（底部x轴）
bee_data = []
for i, category in enumerate(categories):
    n_points = 25
    # 为每个特征生成不同的分布
    base_value = [10, 25, 15, 30, 30][i]
    group_data = np.random.normal(loc=base_value, scale=5, size=n_points)
    for value in group_data:
        bee_data.append({'Category': category, 'Value': value})

bee_df = pd.DataFrame(bee_data)

# 为SHAP值创建数据（顶部x轴）
shap_data = []
for i, category in enumerate(categories):
    n_points = 25
    # SHAP值通常围绕0分布，不同特征有不同的重要性
    base_shap = [0.8, 1.2, 0.3, 1.5, 0.5][i]
    shap_values = np.random.normal(loc=base_shap, scale=0.3, size=n_points)
    for value in shap_values:
        shap_data.append({'Category': category, 'SHAP': value})

shap_df = pd.DataFrame(shap_data)

# 计算每个类别的平均绝对SHAP值用于条形图，并按重要性排序
mean_abs_shap = shap_df.groupby('Category')['SHAP'].apply(lambda x: np.abs(x).mean()).reset_index()
mean_abs_shap.columns = ['Category', 'MeanAbsSHAP']
# 按平均绝对SHAP值降序排序
mean_abs_shap = mean_abs_shap.sort_values('MeanAbsSHAP', ascending=False)
sorted_categories = mean_abs_shap['Category'].tolist()

# 更新数据框中的分类顺序
bee_df['Category'] = pd.Categorical(bee_df['Category'], categories=sorted_categories, ordered=True)
shap_df['Category'] = pd.Categorical(shap_df['Category'], categories=sorted_categories, ordered=True)

# 创建颜色映射对象
from matplotlib.colors import LinearSegmentedColormap

colors = [(0, 'blue'), (1, 'red')]
cmap_name = 'my_colormap'
cm = LinearSegmentedColormap.from_list(cmap_name, colors)

# 创建图形和主坐标轴
fig, ax1 = plt.subplots(figsize=(14, 10))

# 绘制SHAP值散点图（底部x轴 - SHAP值）
sns.stripplot(data=shap_df, x='SHAP', y='Category', ax=ax1,
              palette=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFE66D'],
              size=4, alpha=0.7, jitter=0.2, edgecolor='black', linewidth=0.5,
              marker='s')  # 使用方形标记区分SHAP值，降低size从6到4，jitter从0.3到0.2

# 设置底部x轴的标签和样式（SHAP值）
ax1.set_xlabel('Feature Value Distribution (Bee Swarm)', fontsize=14, fontweight='bold', color='#A23B72')
ax1.tick_params(axis='x', labelcolor='#A23B72')
# ax1.grid(True, alpha=0.3, axis='x', linestyle='--')

# 在SHAP图中添加垂直线标记SHAP=0
ax1.axvline(x=0, color='red', linestyle='-', alpha=0.5, linewidth=1)

# 创建顶部x轴的坐标轴（共享y轴）
ax2 = ax1.twiny()

# 绘制蜂群图（顶部x轴 - 特征值）
# 使用 bee_df 的 Value 字段来映射颜色
norm = plt.Normalize(bee_df['Value'].min(), bee_df['Value'].max())
colors_mapped = cm(norm(bee_df['Value']))

# 将颜色传递给 stripplot
sns.stripplot(data=bee_df, x='Value', y='Category', ax=ax2,
              palette=colors_mapped,
              size=4, alpha=0.7, jitter=0.2, edgecolor='black', linewidth=0.5,
              marker='o')  # 使用圆形标记表示特征值，降低size从6到4，jitter从0.3到0.2

# 设置顶部x轴的标签和样式（特征值分布）
ax2.set_xlabel('SHAP Values (Feature Importance)', fontsize=14, fontweight='bold', color='#2E86AB')
ax2.tick_params(axis='x', labelcolor='#2E86AB')
# ax2.grid(True, alpha=0.3, axis='x')

# 确保顶部x轴从0开始，并与底部x轴对齐
ax2.set_xlim(0, bee_df['Value'].max() + 2)  # 从0开始，扩展到最大值+2

# 获取当前x轴范围来确定条形图位置
x_range = ax2.get_xlim()
# 将条形图起始位置设置为x=0（左侧y轴位置）
bar_start = 0

# 计算条形图的缩放因子 - 降低条形图宽度
max_shap = mean_abs_shap['MeanAbsSHAP'].max()
bar_scale = (x_range[1] - x_range[0]) * 0.10 / max_shap if max_shap > 0 else 1  # 从0.15降低到0.10

# 保存条形图的边界，以便后续设置正确的xlim
bar_ends = []

for i, (category, mean_val) in enumerate(zip(sorted_categories, mean_abs_shap['MeanAbsSHAP'])):
    # 在每个类别旁边绘制一个小的条形图
    # 修复排序顺序问题：直接使用i作为y轴位置，避免反转
    y_pos = i
    bar_width = mean_val * bar_scale
    # 降低条形图高度，从0.6降低到0.4
    rect = plt.Rectangle((bar_start, y_pos - 0.2), bar_width, 0.4, 
                         facecolor='#B3D9FF', alpha=0.3, edgecolor='black', linewidth=0.5)
    ax2.add_patch(rect)
    bar_ends.append(bar_start + bar_width)

# 如果绘制了条形图，则扩展x轴范围以容纳它们
if bar_ends:
    max_bar_end = max(bar_ends)
    new_x_max = max(x_range[1], max_bar_end + (x_range[1] - x_range[0]) * 0.05)
    ax2.set_xlim(x_range[0], new_x_max)

# 设置y轴标签
ax1.set_ylabel('Feature Categories', fontsize=14, fontweight='bold')

# 调整两个x轴的位置
ax1.xaxis.set_ticks_position('bottom')
ax1.xaxis.set_label_position('bottom')
ax2.xaxis.set_ticks_position('top')
ax2.xaxis.set_label_position('top')

# 设置坐标轴范围
ax1.set_xlim(shap_df['SHAP'].min() - 0.5, shap_df['SHAP'].max() + 0.5)
# 对于ax2，我们需要考虑条形图的宽度，所以要动态设置范围
x2_min, x2_max = bee_df['Value'].min() - 2, bee_df['Value'].max() + 2

# 如果绘制了条形图，则扩展x轴范围以容纳它们
if bar_ends:
    max_bar_end = max(bar_ends)
    new_x_max = max(x2_max, max_bar_end + (x2_max - x2_min) * 0.05)
    ax2.set_xlim(x2_min, new_x_max)
else:
    ax2.set_xlim(x2_min, x2_max)

# 添加标题
plt.title('Feature Analysis: Feature Value Distribution vs SHAP Importance', 
          fontsize=16, fontweight='bold', pad=50)

# 添加右侧颜色条
sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
# sm.set_array([])  # only needed for matplotlib < 3.1

# 添加颜色条到右侧
cbar = fig.colorbar(sm, ax=ax1, orientation='vertical', aspect=80)
cbar.ax.set_ylabel('Feature value', rotation=270, labelpad=15, fontsize=14, fontweight='bold')
cbar.ax.text(0.5, 1.05, 'High', transform=cbar.ax.transAxes, ha='center', va='bottom', fontsize=12, fontweight='bold')
cbar.ax.text(0.5, -0.05, 'Low', transform=cbar.ax.transAxes, ha='center', va='top', fontsize=12, fontweight='bold')

# 调整布局
# plt.tight_layout()

# 显示图表
plt.savefig('dual_axis_shap_plot.png', dpi=300, bbox_inches='tight')