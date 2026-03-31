import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
# 添加SHAP库导入
import shap

# 设置英文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def create_dual_axis_shap_plot(bee_df, shap_df, mean_abs_shap, sorted_categories, filename='dual_axis_shap_plot.png'):
    """
    创建双轴SHAP图，包含特征值分布和SHAP值重要性
    
    参数:
    bee_df: 包含特征值分布的DataFrame
    shap_df: 包含SHAP值的DataFrame
    mean_abs_shap: 包含平均绝对SHAP值的DataFrame
    sorted_categories: 按重要性排序的特征类别列表
    filename: 保存图像的文件名
    """
    
    # 更新数据框中的分类顺序
    bee_df['Category'] = pd.Categorical(bee_df['Category'], categories=sorted_categories, ordered=True)
    shap_df['Category'] = pd.Categorical(shap_df['Category'], categories=sorted_categories, ordered=True)

    # 创建颜色映射对象，用于根据SHAP值着色
    # 使用更丰富的颜色梯度，增强视觉效果
    colors = [(0, 'darkblue'), (0.3, 'blue'), (0.5, 'purple'), (0.7, 'red'), (1, 'darkred')]
    cmap_name = 'enhanced_colormap'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors)

    # 创建图形和主坐标轴，使用 constrained_layout 改善布局
    fig, ax1 = plt.subplots(figsize=(14, 10), constrained_layout=False)
    
    # 为每个数据点计算对应的SHAP值，用于着色
    # 计算所有SHAP值的范围用于归一化
    all_shap_values = shap_df['SHAP']
    # 使用实际最小值和最大值进行归一化，避免颜色集中在低端
    shap_min, shap_max = all_shap_values.min(), all_shap_values.max()
    
    # 如果范围太小，适当扩展以增强颜色变化
    if shap_max - shap_min < 1.0:
        shap_max = shap_min + 1.0
    
    # 增加一个安全检查：如果SHAP值范围仍然很小，进一步放大
    if shap_max - shap_min < 2.0:
        # 放大SHAP值以扩展范围
        scale_factor = 2.0
        all_shap_values = all_shap_values * scale_factor
        shap_min, shap_max = all_shap_values.min(), all_shap_values.max()
    
    norm_shap = plt.Normalize(shap_min, shap_max)  # 确保使用真实范围

    # 为bee_df中的每个点根据其SHAP值计算颜色
    point_colors = []
    for i in range(len(bee_df)):
        shap_val = shap_df.iloc[i]['SHAP']
        point_colors.append(cm(norm_shap(shap_val)))

    # 绘制数据点（底部x轴 - 特征值），颜色表示SHAP值重要性
    # 修复hue参数警告：移除hue参数，因为我们手动设置了颜色
    sns.stripplot(data=bee_df, x='Value', y='Category', ax=ax1,
                  color='gray',  # 先设置默认颜色
                  size=4, alpha=0.7, jitter=0.2, edgecolor='black', linewidth=0.5)
                  
    # 然后手动设置每个点的颜色
    for i, artist in enumerate(ax1.collections):
        artist.set_facecolors(point_colors)

    # 设置底部x轴的标签和样式（特征值）
    ax1.set_xlabel('Feature Value Distribution', fontsize=14, fontweight='bold', color='#A23B72')
    ax1.tick_params(axis='x', labelcolor='#A23B72')

    # 在特征值图中添加颜色条表示SHAP值
    sm_shap = plt.cm.ScalarMappable(cmap=cm, norm=norm_shap)

    # 添加颜色条到右侧
    cbar_shap = fig.colorbar(sm_shap, ax=ax1, orientation='vertical', aspect=80)
    cbar_shap.ax.set_ylabel('SHAP Value (Feature Importance)', rotation=270, labelpad=15, fontsize=14, fontweight='bold')

    # 设置颜色条的具体刻度标签，使用真实SHAP值范围
    cbar_shap.ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    actual_min, actual_max = shap_min, shap_max
    cbar_shap.ax.set_yticklabels([
        f'{actual_min:.2f}', 
        f'{actual_min + 0.25*(actual_max-actual_min):.2f}', 
        f'{actual_min + 0.5*(actual_max-actual_min):.2f}', 
        f'{actual_min + 0.75*(actual_max-actual_min):.2f}', 
        f'{actual_max:.2f}'
    ])
    
    cbar_shap.ax.text(0.5, 1.05, 'High', transform=cbar_shap.ax.transAxes, ha='center', va='bottom', fontsize=12, fontweight='bold')
    cbar_shap.ax.text(0.5, -0.05, 'Low', transform=cbar_shap.ax.transAxes, ha='center', va='top', fontsize=12, fontweight='bold')

    # 创建顶部x轴的坐标轴（共享y轴）
    ax2 = ax1.twiny()

    # 绘制透明的SHAP值点（只用于显示SHAP值坐标轴）
    # 移除hue参数，因为不需要着色
    sns.stripplot(data=shap_df, x='SHAP', y='Category', ax=ax2,
                  color='gray',
                  size=4, alpha=0.0, jitter=0.2, edgecolor='black', linewidth=0.5,
                  marker='o')

    # 设置顶部x轴的标签和样式（SHAP值）
    ax2.set_xlabel('SHAP Values', fontsize=14, fontweight='bold', color='#2E86AB')
    ax2.tick_params(axis='x', labelcolor='#2E86AB')

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
    ax1.set_xlim(bee_df['Value'].min() - 2, bee_df['Value'].max() + 2)
    
    # 修改：将顶部x轴的最小值设置为0，避免显示负值
    # 修复：使用实际SHAP值的最大值而不是固定的17.5
    shap_min = max(0, shap_df['SHAP'].min() - 0.5)  # 确保最小值不小于0
    # 使用实际最大值，而不是固定值
    shap_max_actual = shap_df['SHAP'].max() + 0.5
    ax2.set_xlim(shap_min, shap_max_actual)

    # 添加标题
    plt.title('Feature Analysis: Feature Value Distribution vs SHAP Importance', 
              fontsize=16, fontweight='bold', pad=50)

    # 保存图像
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    # 可选：显示图像（仅调试时使用）
    plt.show()


def create_dual_axis_shap_plot_with_summary(physchem_features, shap_values, feature_names, filename='dual_axis_shap_plot_with_summary.png'):
    """
    创建双轴SHAP图，结合SHAP摘要图和特征值分布图
    
    参数:
    physchem_features: 物化特征数据 (numpy array or DataFrame)
    shap_values: SHAP值 (numpy array)
    feature_names: 特征名称列表
    filename: 保存图像的文件名
    """
    
    # 创建图形和子图，使用高度比例分配
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 2]}, constrained_layout=True)
    
    # 1. SHAP摘要图
    plt.sca(axes[0])
    shap.summary_plot(shap_values, physchem_features, 
                     feature_names=feature_names, show=False)
    axes[0].set_title("SHAP Feature Importance Summary")
    
    # 2. 特征值分布图 (保持原有风格)
    # 将数据转换为DataFrame格式
    if isinstance(physchem_features, pd.DataFrame):
        df = physchem_features.copy()
    else:
        df = pd.DataFrame(physchem_features, columns=feature_names)
    
    # 将数据转换为长格式用于绘制
    bee_df = df.melt(var_name='Category', value_name='Value')
    
    # 为SHAP值创建DataFrame
    if isinstance(shap_values, list):  # 多分类情况
        shap_vals = shap_values[0]  # 使用第一个类别的SHAP值
    else:
        shap_vals = shap_values
    
    shap_df = pd.DataFrame(shap_vals, columns=feature_names)
    shap_df = shap_df.melt(var_name='Category', value_name='SHAP')
    
    # 按SHAP值重要性排序特征
    mean_abs_shap = shap_df.groupby('Category')['SHAP'].apply(lambda x: np.abs(x).mean()).reset_index()
    mean_abs_shap.columns = ['Category', 'MeanAbsSHAP']
    mean_abs_shap = mean_abs_shap.sort_values('MeanAbsSHAP', ascending=False)
    sorted_categories = mean_abs_shap['Category'].tolist()
    
    # 更新数据框中的分类顺序
    bee_df['Category'] = pd.Categorical(bee_df['Category'], categories=sorted_categories, ordered=True)
    shap_df['Category'] = pd.Categorical(shap_df['Category'], categories=sorted_categories, ordered=True)

    # 创建颜色映射对象，用于根据SHAP值着色
    colors = [(0, 'darkblue'), (0.3, 'blue'), (0.5, 'purple'), (0.7, 'red'), (1, 'darkred')]
    cmap_name = 'enhanced_colormap'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors)

    # 为每个数据点计算对应的SHAP值，用于着色
    all_shap_values = shap_df['SHAP']
    shap_min, shap_max = all_shap_values.min(), all_shap_values.max()
    
    # 如果范围太小，适当扩展以增强颜色变化
    if shap_max - shap_min < 1.0:
        shap_max = shap_min + 1.0
    
    if shap_max - shap_min < 2.0:
        scale_factor = 2.0
        all_shap_values = all_shap_values * scale_factor
        shap_min, shap_max = all_shap_values.min(), all_shap_values.max()
    
    norm_shap = plt.Normalize(shap_min, shap_max)

    # 为bee_df中的每个点根据其SHAP值计算颜色
    point_colors = []
    for i in range(len(bee_df)):
        shap_val = shap_df.iloc[i]['SHAP']
        point_colors.append(cm(norm_shap(shap_val)))

    # 绘制数据点（x轴 - 特征值），颜色表示SHAP值重要性
    sns.stripplot(data=bee_df, x='Value', y='Category', ax=axes[1],
                  color='gray', size=4, alpha=0.7, jitter=0.2, edgecolor='black', linewidth=0.5)
                  
    # 然后手动设置每个点的颜色
    for i, artist in enumerate(axes[1].collections):
        artist.set_facecolors(point_colors)

    # 设置x轴的标签和样式（特征值）
    axes[1].set_xlabel('Feature Value Distribution', fontsize=14, fontweight='bold', color='#A23B72')
    axes[1].tick_params(axis='x', labelcolor='#A23B72')

    # 添加颜色条表示SHAP值
    sm_shap = plt.cm.ScalarMappable(cmap=cm, norm=norm_shap)
    cbar_shap = fig.colorbar(sm_shap, ax=axes[1], orientation='vertical', aspect=30)
    cbar_shap.ax.set_ylabel('SHAP Value', rotation=270, labelpad=15, fontsize=12, fontweight='bold')

    # 设置颜色条的具体刻度标签
    cbar_shap.ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    actual_min, actual_max = shap_min, shap_max
    cbar_shap.ax.set_yticklabels([
        f'{actual_min:.3f}', 
        f'{actual_min + 0.25*(actual_max-actual_min):.3f}', 
        f'{actual_min + 0.5*(actual_max-actual_min):.3f}', 
        f'{actual_min + 0.75*(actual_max-actual_min):.3f}', 
        f'{actual_max:.3f}'
    ])
    
    # 设置y轴标签
    axes[1].set_ylabel('Feature Categories', fontsize=14, fontweight='bold')
    
    # 添加整体标题
    fig.suptitle('Feature Analysis: SHAP Summary and Feature Value Distribution', 
                 fontsize=16, fontweight='bold')

    # 保存图像
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def generate_sample_data():
    """生成示例数据"""
    # 创建示例数据
    np.random.seed(42)
    categories = ['Feature A', 'Feature B', 'Feature C', 'Feature D', 'Feature E']

    # 为蜂群图创建数据（底部x轴）
    bee_data = []
    for i, category in enumerate(categories):
        n_points = 60  # 增加点的数量
        # 为每个特征生成不同的分布
        base_value = [5, 25, 15, 35, 20][i]
        group_data = np.random.normal(loc=base_value, scale=10, size=n_points)  # 增加标准差使分布更广
        for value in group_data:
            bee_data.append({'Category': category, 'Value': value})

    bee_df = pd.DataFrame(bee_data)

    # 为SHAP值创建数据（顶部x轴）
    shap_data = []
    for i, category in enumerate(categories):
        # 为每个类别生成不同的SHAP值分布，使差异更明显
        n_points = 60
        category_shap_values = []
        
        # 根据类别生成不同的SHAP值分布，扩大范围
        if category == 'Feature A':
            # 低SHAP值分布，但有一定变化
            category_shap_values = np.random.exponential(scale=0.5, size=n_points)
        elif category == 'Feature B':
            # 中等SHAP值分布，但有一定变化
            category_shap_values = np.random.gamma(shape=2, scale=0.8, size=n_points)
        elif category == 'Feature C':
            # 高SHAP值分布
            category_shap_values = np.random.gamma(shape=3, scale=1.2, size=n_points)
        elif category == 'Feature D':
            # 跨大范围的SHAP值分布
            category_shap_values = np.random.exponential(scale=1.5, size=n_points)
        else:  # Feature E
            # 另一种分布，使用混合分布创造更大范围
            if i % 2 == 0:
                category_shap_values = np.random.gamma(shape=2, scale=1.0, size=n_points)
            else:
                category_shap_values = np.random.exponential(scale=0.8, size=n_points)
            
        # 确保SHAP值不为负数（已经是非负的）
        # 适当放大SHAP值以增强颜色差异
        category_shap_values = category_shap_values * 2.0  # 增加倍数以扩大范围
            
        for value in category_shap_values:
            shap_data.append({'Category': category, 'SHAP': value})

    shap_df = pd.DataFrame(shap_data)

    # 计算每个类别的平均绝对SHAP值用于条形图，并按重要性排序
    mean_abs_shap = shap_df.groupby('Category')['SHAP'].apply(lambda x: np.abs(x).mean()).reset_index()
    mean_abs_shap.columns = ['Category', 'MeanAbsSHAP']
    # 按平均绝对SHAP值降序排序
    mean_abs_shap = mean_abs_shap.sort_values('MeanAbsSHAP', ascending=False)
    sorted_categories = mean_abs_shap['Category'].tolist()
    
    return bee_df, shap_df, mean_abs_shap, sorted_categories


def generate_demo_data_with_shap_values():
    """
    根据指定的特征排名和平均SHAP值生成演示数据
    feature_ranking: [
        ["LogP", 0.08601637396533039],
        ["nRotB", 0.026837450592281913],
        ["MW", 0.01668170888181776],
        ["TPSA", 0.01665660213660449]
    ]
    """
    # 定义特征和对应的SHAP值
    features_shap = [
        ("LogP", 0.08601637396533039),
        ("nRotB", 0.026837450592281913),
        ("MW", 0.01668170888181776),
        ("TPSA", 0.01665660213660449)
    ]
    
    # 提取特征名称
    categories = [feature[0] for feature in features_shap]

    # 为蜂群图创建数据（底部x轴）
    bee_data = []
    for i, category in enumerate(categories):
        n_points = 100  # 每个特征100个数据点
        
        # 为每个特征生成不同的分布
        if category == "LogP":
            # LogP通常在-5到10之间
            group_data = np.random.normal(loc=2.5, scale=3, size=n_points)
        elif category == "nRotB":
            # nRotB通常是较小的正整数，这里模拟0-15范围
            group_data = np.random.gamma(shape=2, scale=2, size=n_points)
        elif category == "MW":
            # MW分子量通常在很广的范围，这里模拟200-600
            group_data = np.random.normal(loc=400, scale=100, size=n_points)
        else:  # TPSA
            # TPSA通常在0-200之间
            group_data = np.random.gamma(shape=2, scale=30, size=n_points)
            
        for value in group_data:
            bee_data.append({'Category': category, 'Value': value})

    bee_df = pd.DataFrame(bee_data)

    # 为SHAP值创建数据（顶部x轴）
    shap_data = []
    for i, (category, mean_shap) in enumerate(features_shap):
        n_points = 100
        
        # 根据平均SHAP值生成分布，使数据点围绕平均值分布
        # 使用较小的标准差以保持SHAP值接近平均值
        category_shap_values = np.random.normal(loc=mean_shap, scale=mean_shap*0.3, size=n_points)
        
        # 确保没有负的SHAP值
        category_shap_values = np.abs(category_shap_values)
            
        for value in category_shap_values:
            shap_data.append({'Category': category, 'SHAP': value})

    shap_df = pd.DataFrame(shap_data)

    # 计算每个类别的平均绝对SHAP值用于条形图，并按重要性排序
    mean_abs_shap = shap_df.groupby('Category')['SHAP'].apply(lambda x: np.abs(x).mean()).reset_index()
    mean_abs_shap.columns = ['Category', 'MeanAbsSHAP']
    
    # 按给定顺序排序
    category_order = [feature[0] for feature in features_shap]
    mean_abs_shap['Category'] = pd.Categorical(mean_abs_shap['Category'], categories=category_order, ordered=True)
    mean_abs_shap = mean_abs_shap.sort_values('Category')
    sorted_categories = category_order
    
    return bee_df, shap_df, mean_abs_shap, sorted_categories


if __name__ == "__main__":
    # # 生成示例数据
    # bee_df, shap_df, mean_abs_shap, sorted_categories = generate_sample_data()
    # # 创建并保存图表
    # create_dual_axis_shap_plot(bee_df, shap_df, mean_abs_shap, sorted_categories)
    
    # 生成基于指定SHAP值的演示数据
    bee_df_demo, shap_df_demo, mean_abs_shap_demo, sorted_categories_demo = generate_demo_data_with_shap_values()
    
    # 创建并保存演示图表
    create_dual_axis_shap_plot(bee_df_demo, shap_df_demo, mean_abs_shap_demo, sorted_categories_demo, 
                              filename='dual_axis_shap_plot_demo.png')
    
    # 示例：如何使用新的结合SHAP摘要图的函数
    # 注意：这需要实际的SHAP值和特征数据
    # create_dual_axis_shap_plot_with_summary(physchem_features, shap_values, feature_names,
    #                                        filename='dual_axis_shap_plot_with_summary.png')
