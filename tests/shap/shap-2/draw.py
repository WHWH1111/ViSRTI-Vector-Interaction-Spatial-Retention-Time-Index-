import os
import numpy as np
import matplotlib.pyplot as plt
import shap
from matplotlib.colors import LinearSegmentedColormap

def visualize_shap_results(shap_values, physchem_features, feature_names, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建基础图表和底部轴（散点图用）
    fig, ax_bottom = plt.subplots(1, 1, figsize=(10, 8))
    
    # 创建顶部轴（条形图专用）
    ax_top = ax_bottom.twiny()
    # 共享y轴并反转（匹配SHAP图的特征顺序）
    ax_top.set_ylim(ax_bottom.get_ylim())
    ax_top.invert_yaxis()
    
    # --------------------------
    # 1. 手动计算并绘制条形图（仅用顶部轴）
    # --------------------------
    # 计算特征重要性（SHAP绝对值的均值）
    feature_importance = np.abs(shap_values).mean(0)

    # 获取特征索引（按重要性排序，保持与散点图一致）
    sorted_idx = np.argsort(feature_importance)  # 升序：重要性从低到高
    
    # 绘制条形图（x为重要性值，y为特征位置）
    bars = ax_top.barh(
        y=np.arange(len(feature_names)),  # y轴位置
        width=feature_importance[sorted_idx],  # 条形宽度（重要性值）
        color='lightblue',
        alpha=0.3,
        height=0.6  # 条形高度
    )
    
    # 设置顶部轴属性
    ax_top.set_xlabel("Feature Importance (absolute SHAP value)", labelpad=10, fontsize=10)
    ax_top.set_yticks([])  # 隐藏顶部轴的y刻度
    ax_top.set_xlim(0, max(feature_importance) * 1.1)  # 适配顶部x轴范围
    
    # --------------------------
    # 2. 绘制散点图（底部轴）
    # --------------------------
    plt.sca(ax_bottom)
    shap.summary_plot(
        shap_values, 
        physchem_features, 
        feature_names=feature_names, 
        show=False,
        color_bar=False   # 关闭默认颜色条
    )
    
    # 确保散点图y轴与条形图对齐
    ax_bottom.set_ylim(ax_top.get_ylim())
    ax_bottom.set_xlabel("SHAP value (impact on model output)", labelpad=10, fontsize=10)
    
    # 添加自定义颜色条
    # 定义红蓝渐变（红色在上，蓝色在下）
    colors = [(1, 0, 0), (0, 0, 1)]  # 顶部红色，底部蓝色
    cmap = LinearSegmentedColormap.from_list('shap_cmap', colors)
    
    # 创建颜色条坐标轴（右侧窄条）
    pad, width = 0.02, 0.005
    pos = ax_bottom.get_position()
    cbar_ax = ax_bottom.figure.add_axes([pos.x1 + pad, pos.y0, width, pos.height])
    
    # 生成垂直渐变数据（关键修改：让渐变沿y轴方向）
    # 形状为 (n, 1)，确保渐变是垂直的
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)  # 从上到下对应0到1
    
    # 绘制垂直渐变（aspect='auto'确保填充整个轴）
    cbar_ax.imshow(gradient, aspect='auto', cmap=cmap)
    cbar_ax.set_axis_off()  # 隐藏刻度
    
    # 添加标签
    ax_bottom.figure.text(pos.x1 + pad + width + 0.01, pos.y1, 'High', 
                  va='top', ha='left', fontsize=10)
    ax_bottom.figure.text(pos.x1 + pad + width + 0.01, pos.y0, 'Low', 
                  va='bottom', ha='left', fontsize=10)
    
    # 调整布局
    # plt.tight_layout()
    
    output_path = os.path.join(output_dir, "final_shap_dual_xaxis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path