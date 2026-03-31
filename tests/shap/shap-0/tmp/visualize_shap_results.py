import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_shap_data(shap_dir):
    """
    加载SHAP分析结果数据
    """
    # 加载特征重要性数据
    with open(os.path.join(shap_dir, 'feature_importance.json'), 'r') as f:
        feature_importance = json.load(f)
    
    # 加载详细SHAP数据
    with open(os.path.join(shap_dir, 'shap_details.json'), 'r') as f:
        shap_details = json.load(f)
    
    return feature_importance, shap_details

def create_shap_visualizations(shap_dir, output_dir=None):
    """
    创建SHAP分析结果的可视化图表
    """
    if output_dir is None:
        output_dir = shap_dir
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    feature_importance, shap_details = load_shap_data(shap_dir)
    
    # 提取数据
    feature_names = shap_details['feature_names']
    shap_values = np.array(shap_details['shap_values'])
    physchem_features = np.array(shap_details['physchem_features'])
    mean_abs_shap = np.array(shap_details['mean_abs_shap'])
    
    # 重塑SHAP值为二维数组 (样本数, 特征数)
    if shap_values.ndim == 3:
        shap_values = shap_values.squeeze(axis=-1)  # 去除最后一个维度
    
    # 创建可视化
    create_beeswarm_plot(shap_values, physchem_features, feature_names, output_dir)
    create_feature_importance_plot(feature_importance, output_dir)
    create_combined_plot(shap_values, feature_names, feature_importance, output_dir)

def create_beeswarm_plot(shap_values, physchem_features, feature_names, output_dir):
    """
    创建SHAP蜂巢图(蜜蜂图)
    """
    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 为每个特征创建散点图
    colors = plt.cm.Set1(np.linspace(0, 1, len(feature_names)))
    
    for i, feature in enumerate(feature_names):
        # 获取该特征的SHAP值
        feature_shap = shap_values[:, i]
        
        # 添加一些随机抖动以避免点重叠
        y_jitter = np.random.normal(0, 0.1, len(feature_shap))
        
        # 绘制散点图
        scatter = ax.scatter(feature_shap, 
                           np.full(len(feature_shap), i) + y_jitter,
                           c=[colors[i]], 
                           alpha=0.6, 
                           s=20,
                           label=feature)
    
    # 添加平均值线
    for i, feature in enumerate(feature_names):
        mean_shap = np.mean(shap_values[:, i])
        ax.axvline(mean_shap, ymin=(i-0.4)/len(feature_names), 
                  ymax=(i+0.4)/len(feature_names), 
                  color='red', linestyle='--', linewidth=1)
    
    # 设置标签和标题
    ax.set_xlabel('SHAP Value')
    ax.set_ylabel('Feature')
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names)
    ax.set_title('SHAP Values Distribution (Beeswarm Plot)')
    ax.grid(True, alpha=0.3)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_beeswarm_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_feature_importance_plot(feature_importance, output_dir):
    """
    创建特征重要性条形图
    """
    # 排序特征重要性
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    features, importance = zip(*sorted_features)
    
    # 创建条形图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(features)), importance, color=['red', 'blue', 'green', 'orange'])
    
    # 设置标签和标题
    plt.xlabel('Features')
    plt.ylabel('Mean |SHAP Value|')
    plt.title('Feature Importance based on SHAP Values')
    plt.xticks(range(len(features)), features)
    
    # 在条形上添加数值标签
    for i, (feature, imp) in enumerate(sorted_features):
        plt.text(i, imp + max(importance)*0.01, f'{imp:.4f}', 
                ha='center', va='bottom')
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance_bar_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_combined_plot(shap_values, feature_names, feature_importance, output_dir):
    """
    创建组合图，包含蜂巢图和特征重要性
    """
    # 创建图形和轴
    fig = plt.figure(figsize=(15, 8))
    
    # 创建网格布局
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 左侧：SHAP蜂巢图
    ax1 = fig.add_subplot(gs[:, 0:2])
    
    # 为每个特征创建散点图
    colors = plt.cm.Set1(np.linspace(0, 1, len(feature_names)))
    
    for i, feature in enumerate(feature_names):
        # 获取该特征的SHAP值
        feature_shap = shap_values[:, i]
        
        # 添加一些随机抖动以避免点重叠
        y_jitter = np.random.normal(0, 0.1, len(feature_shap))
        
        # 绘制散点图
        ax1.scatter(feature_shap, 
                   np.full(len(feature_shap), i) + y_jitter,
                   c=[colors[i]], 
                   alpha=0.6, 
                   s=30,
                   label=feature)
    
    # 添加平均值线
    for i, feature in enumerate(feature_names):
        mean_shap = np.mean(shap_values[:, i])
        ax1.axvline(mean_shap, ymin=(i-0.4)/len(feature_names), 
                   ymax=(i+0.4)/len(feature_names), 
                   color='red', linestyle='--', linewidth=1)
    
    # 设置左侧图的标签和标题
    ax1.set_xlabel('SHAP Value')
    ax1.set_ylabel('Feature')
    ax1.set_yticks(range(len(feature_names)))
    ax1.set_yticklabels(feature_names)
    ax1.set_title('SHAP Values Distribution')
    ax1.grid(True, alpha=0.3)
    
    # 右侧：特征重要性条形图
    ax2 = fig.add_subplot(gs[:, 2])
    
    # 排序特征重要性
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    features, importance = zip(*sorted_features)
    
    # 创建水平条形图
    bars = ax2.barh(range(len(features)), importance, color=['red', 'blue', 'green', 'orange'])
    
    # 设置右侧图的标签和标题
    ax2.set_xlabel('Mean |SHAP Value|')
    ax2.set_ylabel('Features')
    ax2.set_yticks(range(len(features)))
    ax2.set_yticklabels(features)
    ax2.set_title('Feature Importance')
    
    # 在条形上添加数值标签
    for i, (feature, imp) in enumerate(sorted_features):
        ax2.text(imp + max(importance)*0.01, i, f'{imp:.4f}', 
                ha='left', va='center')
    
    # 总标题
    fig.suptitle('SHAP Analysis Results', fontsize=16)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_combined_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 设置路径
    shap_dir = '/home/data2/rhj/project/gnn/gnn-1/log/neg-train_20251028-110803_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/shap_analysis'
    
    # 创建可视化
    create_shap_visualizations(shap_dir)
    
    print("SHAP可视化图表已生成并保存到:", shap_dir)

if __name__ == "__main__":
    main()