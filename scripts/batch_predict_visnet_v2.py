#!/usr/bin/env python3
"""
VisNet V2 批量预测脚本
自动遍历模型和对应的fold数据进行批量预测
"""

import os
import subprocess
import argparse
import sys
from pathlib import Path

def get_fold_dirs(data_root):
    """获取所有fold目录"""
    fold_dirs = []
    for item in Path(data_root).iterdir():
        if item.is_dir() and item.name.startswith('fold_'):
            fold_dirs.append(item.name)
    return sorted(fold_dirs)

def get_model_dirs(model_root):
    """获取所有模型目录"""
    model_dirs = []
    for item in Path(model_root).iterdir():
        if item.is_dir() and item.name.startswith('train_'):
            model_dirs.append(item.name)
    return sorted(model_dirs)

def batch_predict(model_root, data_root, ion_type='pos'):
    """
    批量预测函数
    :param model_root: 模型根目录
    :param data_root: 数据根目录
    :param ion_type: 离子类型 ('pos' 或 'neg')
    """
    # 获取所有模型目录
    model_dirs = get_model_dirs(model_root)
    print(f"找到 {len(model_dirs)} 个模型目录")
    
    # 获取所有fold目录
    fold_dirs = get_fold_dirs(data_root)
    print(f"找到 {len(fold_dirs)} 个fold目录")
    
    # 确保模型和fold数量匹配
    if len(model_dirs) != len(fold_dirs):
        print(f"警告: 模型数量({len(model_dirs)})与fold数量({len(fold_dirs)})不匹配")
    
    # 创建输出目录
    output_root = f"./predictions/visnet-v2-5fold/{ion_type}"
    os.makedirs(output_root, exist_ok=True)
    
    # 遍历每个模型和对应的fold
    for i, (model_dir, fold_dir) in enumerate(zip(model_dirs, fold_dirs)):
        print(f"\n处理第 {i+1} 个模型和fold组合:")
        print(f"  模型目录: {model_dir}")
        print(f"  Fold目录: {fold_dir}")
        
        # 构建路径
        model_path = os.path.join(model_root, model_dir, 'best.pt')
        params_path = os.path.join(model_root, model_dir, 'training_params.json')
        dataset_file = os.path.join(data_root, fold_dir, f'{fold_dir}_test_set.txt')
        cache_name = f'visnet-v2-5fold-{ion_type}'
        
        # 检查必要文件是否存在
        if not os.path.exists(model_path):
            print(f"  跳过: 模型文件不存在 {model_path}")
            continue
            
        if not os.path.exists(params_path):
            print(f"  跳过: 参数文件不存在 {params_path}")
            continue
            
        if not os.path.exists(dataset_file):
            print(f"  跳过: 数据集文件不存在 {dataset_file}")
            continue
        
        # 构建命令
        cmd = [
            sys.executable, 'predict_visnet_v2.py',
            '--model_path', model_path,
            '--params_path', params_path,
            '--input_file', f'data\\MMF-3\\MMF_GNN_{ion_type}.csv',
            '--output_dir', output_root,
            '--batch_size', '64',
            '--dataset_file', dataset_file,
            '--cache_name', cache_name
        ]
        
        # 执行命令
        print(f"  执行命令: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            if result.returncode == 0:
                print(f"  成功完成预测")
            else:
                print(f"  预测失败: {result.stderr}")
        except Exception as e:
            print(f"  执行异常: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='VisNet V2 批量预测脚本')
    parser.add_argument('--model_root', type=str, 
                        default='log/visnet-v2-5/pos', 
                        help='模型根目录 (default: log/visnet-v2-5/pos)')
    parser.add_argument('--data_root', type=str, 
                        default='data/MMF-3/MMF_GNN_pos_5', 
                        help='数据根目录 (default: data/MMF-3/MMF_GNN_pos_5)')
    parser.add_argument('--ion_type', type=str, choices=['pos', 'neg'], 
                        default='pos', help='离子类型 (default: pos)')
    
    args = parser.parse_args()
    
    print("VisNet V2 批量预测脚本")
    print("=" * 50)
    print(f"模型根目录: {args.model_root}")
    print(f"数据根目录: {args.data_root}")
    print(f"离子类型: {args.ion_type}")
    print("=" * 50)
    
    batch_predict(args.model_root, args.data_root, args.ion_type)

if __name__ == '__main__':
    main()