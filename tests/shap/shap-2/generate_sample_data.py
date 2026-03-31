import numpy as np
import pandas as pd
import os
import sys

# 添加当前目录到Python路径，以便导入draw模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入可视化函数
try:
    import draw
    HAS_DRAW_MODULE = True
except ImportError as e:
    HAS_DRAW_MODULE = False
    print(f"无法导入draw模块: {e}")
    print("将只生成数据而不进行可视化")

def generate_sample_data(n_samples=500):
    """
    生成用于SHAP可视化的示例数据
    
    Args:
        n_samples: 样本数量
        
    Returns:
        tuple: (shap_values, physchem_features, feature_names)
    """
    # 定义特征名称（按重要性排序：LogP > nRotB > MW > TPSA）
    feature_names = [
        'LogP',
        'MW',
        'TPSA',
        'nRotB'
    ]
    
    n_features = len(feature_names)
    
    # 生成物理化学特征数据
    # 使用不同的分布来模拟真实世界中的特征
    physchem_features = np.zeros((n_samples, n_features))
    
    # LogP - 正态分布，范围通常在-5到5之间
    physchem_features[:, 0] = np.random.normal(2.0, 1.5, n_samples)
    
    # MW (分子量) - 正态分布，药物分子通常在200-600之间
    physchem_features[:, 1] = np.random.normal(400, 100, n_samples)
    
    # TPSA (拓扑极性表面积) - 指数分布，多数在0-150之间
    physchem_features[:, 2] = np.random.exponential(40, n_samples)
    
    # nRotB (可旋转键数量) - 泊松分布，通常在0-15之间
    physchem_features[:, 3] = np.random.poisson(4, n_samples)
    
    # 生成SHAP值 - 基于特征重要性排序
    shap_values = np.zeros((n_samples, n_features))
    
    # 根据给定的重要性排序生成SHAP值
    # LogP是最重要的特征，mean_abs_shap = 0.086
    shap_values[:, 0] = np.random.normal(0, 0.086*0.6, n_samples) + np.random.uniform(-0.02, 0.02, n_samples)
    
    # nRotB 第二重要，mean_abs_shap = 0.027
    shap_values[:, 3] = np.random.normal(0, 0.027*0.6, n_samples) + np.random.uniform(-0.01, 0.01, n_samples)
    
    # MW 第三重要，mean_abs_shap = 0.017
    shap_values[:, 1] = np.random.normal(0, 0.017*0.6, n_samples) + np.random.uniform(-0.005, 0.005, n_samples)
    
    # TPSA 最不重要，mean_abs_shap = 0.017
    shap_values[:, 2] = np.random.normal(0, 0.017*0.6, n_samples) + np.random.uniform(-0.005, 0.005, n_samples)
    
    # 添加一些随机性以增加多样性
    noise = np.random.normal(0, 0.005, (n_samples, n_features))
    shap_values = shap_values + noise
    
    return shap_values, physchem_features, feature_names

def save_sample_data_to_csv(shap_values, physchem_features, feature_names, output_dir="./sample_data"):
    """
    将生成的示例数据保存为CSV文件
    
    Args:
        shap_values: SHAP值数组
        physchem_features: 物化特征数组
        feature_names: 特征名称列表
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存特征数据
    feature_df = pd.DataFrame(physchem_features, columns=feature_names)
    feature_df.to_csv(os.path.join(output_dir, "physchem_features.csv"), index=False)
    
    # 保存SHAP值数据
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_df.to_csv(os.path.join(output_dir, "shap_values.csv"), index=False)
    
    print(f"示例数据已保存到 {output_dir} 目录")

if __name__ == "__main__":
    # 生成示例数据
    shap_values, physchem_features, feature_names = generate_sample_data(n_samples=500)
    
    # 保存数据到CSV文件
    save_sample_data_to_csv(shap_values, physchem_features, feature_names)
    
    # 如果draw.py中的函数可用，则运行可视化
    if HAS_DRAW_MODULE:
        try:
            draw.visualize_shap_results(shap_values, physchem_features, feature_names, "./shap_visualizations")
        except Exception as e:
            print(f"可视化过程中出现错误: {e}")
            print("请确保draw.py中的函数正确且所需依赖已安装")
    else:
        print("draw模块不可用，跳过可视化步骤")