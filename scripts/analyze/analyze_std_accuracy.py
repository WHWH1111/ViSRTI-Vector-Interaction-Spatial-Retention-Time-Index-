import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

def analyze_confidence_correlation(csv_file_path):
    """
    分析置信度与预测误差之间的相关性
    """
    # 读取数据
    df = pd.read_csv(csv_file_path)
    
    # 计算绝对误差
    df['abs_error'] = np.abs(df['Predicted'] - df['Actual'])
    
    # 计算相对误差（百分比）
    df['rel_error'] = (df['abs_error'] / df['Actual']) * 100
    
    print("=== 数据概览 ===")
    print(f"总样本数: {len(df)}")
    print(f"置信度范围: {df['Confidence'].min():.3f} - {df['Confidence'].max():.3f}")
    print(f"绝对误差范围: {df['abs_error'].min():.3f} - {df['abs_error'].max():.3f}")
    print(f"标准差范围: {df['Std'].min():.3f} - {df['Std'].max():.3f}")
    
    # 计算相关性
    corr_abs_confidence = pearsonr(df['abs_error'], df['Confidence'])
    corr_rel_confidence = pearsonr(df['rel_error'], df['Confidence']) 
    corr_abs_std = pearsonr(df['abs_error'], df['Std'])
    
    # 斯皮尔曼秩相关（对异常值更鲁棒）
    spear_abs_confidence = spearmanr(df['abs_error'], df['Confidence'])
    spear_rel_confidence = spearmanr(df['rel_error'], df['Confidence'])
    spear_abs_std = spearmanr(df['abs_error'], df['Std'])
    
    print("\n=== 相关性分析 ===")
    print("置信度 vs 绝对误差:")
    print(f"  皮尔逊相关系数: {corr_abs_confidence[0]:.3f} (p-value: {corr_abs_confidence[1]:.3f})")
    print(f"  斯皮尔曼相关系数: {spear_abs_confidence[0]:.3f} (p-value: {spear_abs_confidence[1]:.3f})")
    
    print("\n置信度 vs 相对误差:")
    print(f"  皮尔逊相关系数: {corr_rel_confidence[0]:.3f} (p-value: {corr_rel_confidence[1]:.3f})")
    print(f"  斯皮尔曼相关系数: {spear_rel_confidence[0]:.3f} (p-value: {spear_rel_confidence[1]:.3f})")
    
    print("\n标准差 vs 绝对误差:")
    print(f"  皮尔逊相关系数: {corr_abs_std[0]:.3f} (p-value: {corr_abs_std[1]:.3f})")
    print(f"  斯皮尔曼相关系数: {spear_abs_std[0]:.3f} (p-value: {spear_abs_std[1]:.3f})")
    
    # # 创建可视化
    # fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # # 1. 置信度 vs 绝对误差
    # axes[0,0].scatter(df['Confidence'], df['abs_error'], alpha=0.6)
    # axes[0,0].set_xlabel('Confidence')
    # axes[0,0].set_ylabel('Absolute Error')
    # axes[0,0].set_title(f'Confidence vs Absolute Error\n(Pearson: {corr_abs_confidence[0]:.3f})')
    
    # # 2. 置信度 vs 相对误差
    # axes[0,1].scatter(df['Confidence'], df['rel_error'], alpha=0.6)
    # axes[0,1].set_xlabel('Confidence')
    # axes[0,1].set_ylabel('Relative Error (%)')
    # axes[0,1].set_title(f'Confidence vs Relative Error\n(Pearson: {corr_rel_confidence[0]:.3f})')
    
    # # 3. 标准差 vs 绝对误差
    # axes[0,2].scatter(df['Std'], df['abs_error'], alpha=0.6)
    # axes[0,2].set_xlabel('Standard Deviation')
    # axes[0,2].set_ylabel('Absolute Error')
    # axes[0,2].set_title(f'Std vs Absolute Error\n(Pearson: {corr_abs_std[0]:.3f})')
    
    # # 4. 置信度分布
    # axes[1,0].hist(df['Confidence'], bins=20, alpha=0.7, edgecolor='black')
    # axes[1,0].set_xlabel('Confidence')
    # axes[1,0].set_ylabel('Frequency')
    # axes[1,0].set_title('Confidence Distribution')
    
    # # 5. 误差分布
    # axes[1,1].hist(df['abs_error'], bins=20, alpha=0.7, edgecolor='black')
    # axes[1,1].set_xlabel('Absolute Error')
    # axes[1,1].set_ylabel('Frequency')
    # axes[1,1].set_title('Absolute Error Distribution')
    
    # # 6. 可靠性图 (Reliability Diagram)
    # # 将置信度分箱，计算每个箱的平均误差
    # df_sorted = df.sort_values('Confidence')
    # n_bins = 10
    # bins = np.array_split(df_sorted, n_bins)
    
    # bin_confidence = []
    # bin_avg_error = []
    
    # for i, bin_df in enumerate(bins):
    #     if len(bin_df) > 0:
    #         bin_confidence.append(bin_df['Confidence'].mean())
    #         bin_avg_error.append(bin_df['abs_error'].mean())
    
    # axes[1,2].plot(bin_confidence, bin_avg_error, 'o-', linewidth=2)
    # axes[1,2].plot([0, 1], [0, max(bin_avg_error)], '--', alpha=0.5, color='red')
    # axes[1,2].set_xlabel('Average Confidence in Bin')
    # axes[1,2].set_ylabel('Average Absolute Error in Bin')
    # axes[1,2].set_title('Reliability Diagram')
    # axes[1,2].grid(True, alpha=0.3)
    
    # plt.tight_layout()
    # plt.savefig('confidence_analysis.png', dpi=300, bbox_inches='tight')
    # # plt.show()
    
    # 分析高/低置信度样本的表现
    high_conf_threshold = df['Confidence'].quantile(0.75)  # 前25%
    low_conf_threshold = df['Confidence'].quantile(0.25)   # 后25%
    
    high_conf_samples = df[df['Confidence'] >= high_conf_threshold]
    low_conf_samples = df[df['Confidence'] <= low_conf_threshold]
    
    print("\n=== 高/低置信度样本对比 ===")
    print(f"高置信度阈值: {high_conf_threshold:.3f} (样本数: {len(high_conf_samples)})")
    print(f"低置信度阈值: {low_conf_threshold:.3f} (样本数: {len(low_conf_samples)})")
    
    print(f"\n高置信度样本:")
    print(f"  平均绝对误差: {high_conf_samples['abs_error'].mean():.3f}")
    print(f"  平均相对误差: {high_conf_samples['rel_error'].mean():.3f}%")
    print(f"  平均标准差: {high_conf_samples['Std'].mean():.3f}")
    
    print(f"\n低置信度样本:")
    print(f"  平均绝对误差: {low_conf_samples['abs_error'].mean():.3f}")
    print(f"  平均相对误差: {low_conf_samples['rel_error'].mean():.3f}%")
    print(f"  平均标准差: {low_conf_samples['Std'].mean():.3f}")
    
    # 计算改进比例
    error_improvement = (low_conf_samples['abs_error'].mean() - high_conf_samples['abs_error'].mean()) / low_conf_samples['abs_error'].mean() * 100
    print(f"\n高置信度样本相比低置信度样本误差降低: {error_improvement:.1f}%")
    
    return df

# 使用示例
if __name__ == "__main__":
    csv_file = r"D:\Projects\python\gnn-rt-1\predictions\with-conf\mc_dropout_prediction_20251127-025102\test_MMF_GNN_neg_mc_dropout_predictions.csv"
    df_analysis = analyze_confidence_correlation(csv_file)