import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import pandas as pd

def calculate_metrics(file_path):
    """
    计算预测结果的评估指标
    
    Parameters:
    file_path (str): 包含SMILES、Correct和Predict列的文件路径
    """
    # 读取数据
    df = pd.read_csv(file_path, sep='\t')
    
    # 提取真实值和预测值
    y_true = df['Correct'].values
    y_pred = df['Predict'].values
    
    # 计算评估指标
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    pcc, _ = pearsonr(y_true, y_pred)
    
    # 打印结果
    print("评估指标结果:")
    print(f"样本数量: {len(y_true)}")
    print(f"MAE (Mean Absolute Error): {mae:.4f}")
    print(f"MSE (Mean Squared Error): {mse:.4f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"R² (Coefficient of Determination): {r2:.4f}")
    print(f"PCC (Pearson Correlation Coefficient): {pcc:.4f}")
    
    # 返回结果字典
    return {
        'count': len(y_true),
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'pcc': pcc
    }

if __name__ == "__main__":
    import sys
    
    # 默认文件路径
    file_path = "log/train_20251121-154235_dim48_layerH6_layerO6_batch64_lr0.0001_iter50/train_prediction_train.txt"
    
    # 如果提供了命令行参数，则使用该参数作为文件路径
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    
    try:
        metrics = calculate_metrics(file_path)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
    except Exception as e:
        print(f"计算评估指标时发生错误: {e}")