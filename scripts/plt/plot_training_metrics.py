# > plot_training_metrics.py


import json
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def denormalize_mae(mae_normalized, std):
    """
    将标准化后的MAE转换为原始尺度
    MAE是绝对误差的平均值，因此需要乘以标准差来还原
    """
    return mae_normalized / std

def denormalize_mse(mse_normalized, std):
    """
    将标准化后的MSE转换为原始尺度
    MSE是平方误差的平均值，因此需要乘以标准差的平方来还原
    """
    return mse_normalized / (std ** 2)

def plot_training_metrics(log_dir):
    """
    读取训练指标数据，将其还原为原始值并绘制图表
    """
    # 读取训练参数获取均值和标准差
    params_file = os.path.join(log_dir, 'training_params.json')
    with open(params_file, 'r') as f:
        params = json.load(f)
    
    property_mean = params['property_mean']
    property_std = params['property_std']
    
    print(f"Property mean: {property_mean}")
    print(f"Property std: {property_std}")
    
    # 读取训练指标数据
    metrics_file = os.path.join(log_dir, 'training_metrics.json')
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    mae_train = np.array(metrics['MAE_train'])
    mse_train = np.array(metrics['MSE_train'])
    mae_val = np.array(metrics['MAE_dev'])
    mse_val = np.array(metrics['MSE_dev'])
    
    # 将标准化指标还原为原始值
    mae_train_original = denormalize_mae(mae_train, property_std)
    mse_train_original = denormalize_mse(mse_train, property_std)
    mae_val_original = denormalize_mae(mae_val, property_std)
    mse_val_original = denormalize_mse(mse_val, property_std)
    
    # 将处理后的数据保存为JSON文件
    processed_metrics = {
        "MAE_train_original": mae_train_original.tolist(),
        "MSE_train_original": mse_train_original.tolist(),
        "MAE_val_original": mae_val_original.tolist(),
        "MSE_val_original": mse_val_original.tolist(),
        "RMSE_train_original": np.sqrt(mse_train_original).tolist(),
        "RMSE_val_original": np.sqrt(mse_val_original).tolist()
    }
    
    processed_metrics_file = os.path.join(log_dir, 'training_metrics_processed.json')
    with open(processed_metrics_file, 'w') as f:
        json.dump(processed_metrics, f, indent=2)
    
    print(f"Processed metrics saved to: {processed_metrics_file}")
    
    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    epochs = range(1, len(mae_train_original) + 1)
    
    # 绘制MAE图表，包含训练集和验证集的对比
    axes[0].plot(epochs, mae_train_original, label='Train MAE', color='blue')
    axes[0].plot(epochs, mae_val_original, label='Validation MAE', color='orange')
    axes[0].set_title('Mean Absolute Error (MAE)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MAE')
    axes[0].legend()
    axes[0].grid(True)
    
    # 绘制MSE图表，包含训练集和验证集的对比
    axes[1].plot(epochs, mse_train_original, label='Train MSE', color='blue')
    axes[1].plot(epochs, mse_val_original, label='Validation MSE', color='orange')
    axes[1].set_title('Mean Squared Error (MSE)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MSE')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'training_metrics_original_scale.png'), dpi=300, bbox_inches='tight')
    # plt.show()
    
    # 打印最终的指标值
    print(f"Final Train MAE (original scale): {mae_train_original[-1]:.2f}")
    print(f"Final Validation MAE (original scale): {mae_val_original[-1]:.2f}")
    print(f"Final Train MSE (original scale): {mse_train_original[-1]:.2f}")
    print(f"Final Validation MSE (original scale): {mse_val_original[-1]:.2f}")
    
    return processed_metrics

if __name__ == "__main__":
    dir_name = r"neg-3-mask(kow)-train_20251121-171553_dim48_layerH6_layerO6_batch64_lr0.0001_iter150"
    # 指定日志目录
    log_directory = os.path.join(r"log\data-end", dir_name)
    
    # 绘制训练指标
    processed_data = plot_training_metrics(log_directory)