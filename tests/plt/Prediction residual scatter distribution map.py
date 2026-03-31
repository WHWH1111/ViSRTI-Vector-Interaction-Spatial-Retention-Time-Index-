import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ----------------------
# 1. 生成随机数据（线性关系 + 随机噪声）
# ----------------------
np.random.seed(42)  # 固定随机种子，保证结果可复现
X = np.linspace(0, 10, 100).reshape(-1, 1)  # 自变量 X（100 个样本）
y_true = 2 * X.ravel() + 3 + np.random.normal(0, 1, size=X.ravel().shape)  # 真实值：y = 2X + 3 + 随机噪声
y_pred = y_true + np.random.normal(0, 1.5, size=y_true.shape)  # 预测值：真实值 + 更多噪声

# ----------------------
# 2. 计算残差
# ----------------------
residuals = y_true - y_pred

# ----------------------
# 3. 绘制预测残差散点分布图
# ----------------------
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文（Windows）
plt.rcParams['axes.unicode_minus'] = False  # 支持负号

# 创建画布
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# 子图 1：残差 vs 预测值散点图（核心）
ax1.scatter(y_pred, residuals, alpha=0.6, color='#1f77b4', edgecolors='white', s=50)
ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Residual=0')  # 参考线
ax1.set_xlabel('Predicted Values', fontsize=12)
ax1.set_ylabel('Residuals', fontsize=12)
ax1.set_title('Predicted Residual Scatter Plot (Residuals vs Predicted Values)', fontsize=14, fontweight='bold', pad=20)
ax1.legend()
ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax1.set_facecolor('#f8f9fa')  # 背景色

# 子图 2：残差直方图（辅助查看残差分布）
ax2.hist(residuals, bins=15, color='#ff7f0e', alpha=0.7, edgecolor='black', linewidth=0.8)
ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Residual=0')  # 参考线
ax2.set_xlabel('Residuals', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.set_title('Residual Distribution Histogram', fontsize=14, fontweight='bold', pad=20)
ax2.legend()
ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')
ax2.set_facecolor('#f8f9fa')  # 背景色

# 调整布局
plt.tight_layout()
plt.subplots_adjust(top=0.92)  # 顶部留白，避免标题被截断

# 添加整体标题
fig.suptitle(f'Residual Analysis (R²={r2_score(y_true, y_pred):.3f})', 
             fontsize=16, fontweight='bold', y=0.98)

# 保存图片（可选）
plt.savefig('predicted_residual_scatter_distribution.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

def load_prediction_data(filepath):
    """
    加载预测数据
    
    Args:
        filepath (str): 预测结果文件路径
        
    Returns:
        tuple: (y_true, y_pred) 真实值和预测值数组
    """
    # 检查文件是否存在
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"预测结果文件不存在: {filepath}")
    
    # 读取数据，根据项目中的格式，使用制表符分隔
    try:
        data = pd.read_csv(filepath, sep='\t')
    except Exception as e:
        print(f"读取文件时出错: {e}")
        raise
    
    # 根据项目中calculate_metrics.py的实现，列名为'Correct'和'Predict'
    if 'Correct' in data.columns and 'Predict' in data.columns:
        y_true = data['Correct'].values
        y_pred = data['Predict'].values
    else:
        raise ValueError(f"文件 {filepath} 中未找到预期的列 'Correct' 和 'Predict'")
    
    # 检查并处理NaN或无穷大值
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.all(valid_mask):
        print(f"警告: 发现 {np.sum(~valid_mask)} 个无效值，将被忽略")
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
    
    return y_true, y_pred

def plot_residuals(y_true, y_pred, output_file='预测残差散点分布图.png'):
    """
    绘制预测残差散点分布图
    
    Args:
        y_true (array): 真实值
        y_pred (array): 预测值
        output_file (str): 输出文件名
    """
    # 计算残差
    residuals = y_true - y_pred
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建画布
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # 子图 1：残差 vs 预测值散点图
    ax1.scatter(y_pred, residuals, alpha=0.6, color='#1f77b4', edgecolors='white', s=50)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, label='残差=0')
    ax1.set_xlabel('预测值', fontsize=12)
    ax1.set_ylabel('残差', fontsize=12)
    ax1.set_title('预测残差散点图（残差 vs 预测值）', fontsize=14, fontweight='bold', pad=20)
    ax1.legend()
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.set_facecolor('#f8f9fa')

    # 子图 2：残差直方图
    ax2.hist(residuals, bins=15, color='#ff7f0e', alpha=0.7, edgecolor='black', linewidth=0.8)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='残差=0')
    ax2.set_xlabel('残差', fontsize=12)
    ax2.set_ylabel('频数', fontsize=12)
    ax2.set_title('残差分布直方图', fontsize=14, fontweight='bold', pad=20)
    ax2.legend()
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')
    ax2.set_facecolor('#f8f9fa')

    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    # 添加整体标题
    r2 = r2_score(y_true, y_pred)
    fig.suptitle(f'预测残差分析 (R²={r2:.3f})', 
                 fontsize=16, fontweight='bold', y=0.98)

    # 保存图片
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"残差分析图已保存至: {output_file}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='绘制预测残差散点分布图')
    parser.add_argument('--input', '-i', type=str, required=True, help='预测结果文件路径')
    parser.add_argument('--output', '-o', type=str, default='预测残差散点分布图.png', help='输出图像文件路径')
    
    args = parser.parse_args()
    
    try:
        # 加载数据
        y_true, y_pred = load_prediction_data(args.input)
        
        # 检查是否有足够的数据
        if len(y_true) == 0:
            print("错误: 没有有效的预测数据")
            return
        
        print(f"加载了 {len(y_true)} 个有效预测样本")
        
        # 绘制残差图
        plot_residuals(y_true, y_pred, args.output)
        
    except Exception as e:
        print(f"处理过程中出错: {e}")

# 当直接运行脚本时的行为
if __name__ == '__main__':
    if len(sys.argv) == 1:
        # 显示帮助信息
        print("用法: python \"预测残差散点分布图.py\" -i <预测结果文件> [-o <输出图像文件>]")
        print("示例: python \"预测残差散点分布图.py\" -i predictions.txt -o residual_plot.png")
        print("\n如果没有提供参数，显示此帮助信息")
    else:
        main()