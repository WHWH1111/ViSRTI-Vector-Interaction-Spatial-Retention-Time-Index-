import numpy as np

def std_to_confidence_sigmoid(std, sensitivity=10000):
    """
    使用Sigmoid函数将Std映射到0-1的置信度
    sensitivity: 控制函数的敏感度，值越大对小的Std越敏感
    """
    # Sigmoid函数：1 / (1 + exp(-k*(1/std - threshold)))
    # 调整参数使得小Std对应高置信度
    x = 1 / (std + 1e-8)  # Std越小，x越大
    confidence = 1 / (1 + np.exp(-sensitivity * (x - 0.0001)))
    return confidence

# 应用到您的数据
std_values = [2.6490753095022777e-05, 3.328828622222372e-05]
confidences = [std_to_confidence_sigmoid(std) for std in std_values]

for i, (std, conf) in enumerate(zip(std_values, confidences)):
    print(f"样本{i+1}: Std = {std:.2e} -> 置信度 = {conf:.4f}")