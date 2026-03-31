import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from visnet_v2_for_gendata import load_model, load_and_process_dataset_with_features, predict_dataset
import csv
import datetime
import json as json_module
import numpy as np


def main():
    # 更新路径以适应新的文件位置
    path = "../../data/MMF-3"
    
    # 导入预测所需模块
    try:
        # 准备预测参数
        class Args:
            def __init__(self, model_path, params_path, input_file, output_dir, batch_size, dataset_type=None):
                self.model_path = model_path
                self.params_path = params_path
                self.input_file = input_file
                self.output_dir = output_dir
                self.batch_size = batch_size
                self.dataset_type = dataset_type
                self.filter_column = None
                self.filter_value = None
        
        # 创建预测参数对象
        predict_args = Args(
            model_path=None,
            params_path=None,
            input_file=os.path.join(path, 'data1028_sheet0.csv'),  # INFO ori
            output_dir=None,
            batch_size=128,
            # dataset_type='test',
        )
        
        # 检查测试数据集文件是否存在
        if not os.path.exists(predict_args.input_file):
            print(f"Warning: Test dataset file {predict_args.input_file} not found. Skipping automatic prediction.")
        else:
            print(f"Processing {predict_args.dataset_type} dataset for prediction...")
            dataset = load_and_process_dataset_with_features(
                predict_args.input_file, 
                {
                    "dataname": 'MMF-3-neg',
                    'visnet_v2_feature_level': 'graph_physchem_toxicity',
                },
                predict_args.dataset_type,
                # max_data=200
            )

    except Exception as e:
        print(f"Warning: Could not perform automatic prediction: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()