import sys
import os

from matplotlib.dates import MO
# 获取当前文件的目录并将其添加到sys.path中，以确保可以进行绝对导入
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    import timeit
    import numpy as np
    import torch
    import torch.optim as optim
    from tqdm import tqdm
    import os
    import argparse

    # 导入工具函数
    from utils.plot_utils import plot_training_metrics, plot_predicted_vs_actual
    from utils.log_utils import (print_training_setup_info, print_preprocess_info, 
                                print_model_info, print_training_start_info,
                                print_training_time_estimate, print_training_header,
                                print_early_stopping_info, create_log_dir,
                                save_training_params)
    from utils.feature_utils import get_feature_config

    # 导入模型类
    from models.base_model import MolecularGraphNeuralNetwork
    # from models.dual_output_model import DualOutputMolecularGraphNeuralNetwork
    from models.extended_model import ExtendedMolecularGraphNeuralNetwork
    from models.visnet import ViSNet, VisNetV1
    from models.visnet_v2 import VisNetV2

    # 导入训练器和测试器以及数据预处理模块
    from core import (
        Trainer, 
        Tester, 
        split_raw_data_if_needed,
        load_datasets,
        standardize_datasets_if_needed,
        preprocess_visnet_data
    )
except ImportError as e:
    print(f"Error importing modules: {e}")

# 在训练开始前，确保使用固定字典
def load_fixed_dictionaries(path, dataname):
    """在训练开始前加载固定字典"""
    dict_files = {
        'atom': f'{path}/{dataname}/{dataname}-atom_dict.pickle',
        'bond': f'{path}/{dataname}/{dataname}-bond_dict.pickle',
        'fingerprint': f'{path}/{dataname}/{dataname}-fingerprint_dict.pickle',
        'edge': f'{path}/{dataname}/{dataname}-edge_dict.pickle'
    }
    
    for dict_name, file_path in dict_files.items():
        if os.path.exists(file_path):
            print(f"Found existing {dict_name} dictionary: {file_path}")
        else:
            print(f"Note: {dict_name} dictionary file {file_path} does not exist yet.")

_visnet_train = "visnet_train_"
_visnet_test = "visnet_test_"

# 已将Trainer和Tester类移到core/trainer_tester.py文件中

def get_model(args, device, N, dim, layer_hidden, layer_output, additional_features_dim):
    """根据命令行参数选择并初始化模型"""
    if args.model == 'visnet':
        model = ViSNet(
            hidden_channels=args.visnet_hidden_channels,
            num_layers=args.visnet_num_layers,
            num_heads=args.visnet_num_heads,
            num_rbf=args.visnet_num_rbf,
            cutoff=args.visnet_cutoff
        ).to(device)
        print(f"Using VisNet model with hidden_channels={args.visnet_hidden_channels}, "
              f"num_layers={args.visnet_num_layers}, num_heads={args.visnet_num_heads}, "
              f"num_rbf={args.visnet_num_rbf}, cutoff={args.visnet_cutoff}")
    elif args.model == 'visnet_v2':
        # 创建 VisNetV2 模型实例
        model = VisNetV2(
            node_feature_dim=args.visnet_v2_node_feature_dim,
            physchem_feature_dim=args.visnet_v2_physchem_feature_dim,
            toxicity_feature_dim=args.visnet_v2_toxicity_feature_dim,
            chromato_feature_dim=args.visnet_v2_chromato_feature_dim,
            graph_hidden_dim=args.visnet_v2_graph_hidden_dim,
            physchem_hidden_dim=args.visnet_v2_physchem_hidden_dim,
            toxicity_hidden_dim=args.visnet_v2_toxicity_hidden_dim,
            chromato_hidden_dim=args.visnet_v2_chromato_hidden_dim,
            fusion_hidden_dims=args.visnet_v2_fusion_hidden_dims,
            dropout_rate=args.visnet_v2_dropout_rate,
            use_attention=args.visnet_v2_use_attention,
            use_gating=args.visnet_v2_use_gating,
            feature_level=args.visnet_v2_feature_level,  # 将 feature_level 参数传递给模型
            physchem_feature_mask=args.visnet_v2_physchem_mask,  # 将特征掩码参数传递给模型
            toxicity_feature_mask=args.visnet_v2_toxicity_mask,  # 将毒性特征掩码参数传递给模型
            chromato_feature_mask=args.visnet_v2_chromato_mask   # 将色谱特征掩码参数传递给模型
        )
        
        # 添加特征级别属性以便在训练器中使用
        model.feature_level = args.visnet_v2_feature_level
        
        model.to(device)
        print(f"Using VisNetV2 model with node_feature_dim={args.visnet_v2_node_feature_dim}, "
              f"physchem_feature_dim={args.visnet_v2_physchem_feature_dim}, "
              f"toxicity_feature_dim={args.visnet_v2_toxicity_feature_dim}, "
              f"chromato_feature_dim={args.visnet_v2_chromato_feature_dim}, "
              f"graph_hidden_dim={args.visnet_v2_graph_hidden_dim}, "
              f"physchem_hidden_dim={args.visnet_v2_physchem_hidden_dim}, "
              f"toxicity_hidden_dim={args.visnet_v2_toxicity_hidden_dim}, "
              f"chromato_hidden_dim={args.visnet_v2_chromato_hidden_dim}, "
              f"fusion_hidden_dims={args.visnet_v2_fusion_hidden_dims}, "
              f"num_layers={args.visnet_v2_num_layers}, "
              f"num_heads={args.visnet_v2_num_heads}, "
              f"dropout_rate={args.visnet_v2_dropout_rate}, "
              f"use_attention={args.visnet_v2_use_attention}, "
              f"use_gating={args.visnet_v2_use_gating}, "
              f"feature_level={args.visnet_v2_feature_level}, "
              f"physchem_feature_mask={args.visnet_v2_physchem_mask}, "
              f"toxicity_feature_mask={args.visnet_v2_toxicity_mask}, "
              f"chromato_feature_mask={args.visnet_v2_chromato_mask}")

    elif args.model == 'visnet_v1':
        model = VisNetV1(
            node_feature_dim=args.visnet_mfe_node_feature_dim,
            hidden_dims=args.visnet_mfe_hidden_dims,
            output_dims=args.visnet_mfe_output_dims,
            dropout_rate=args.dropout  # 使用统一的dropout参数
        ).to(device)
        print(f"Using VisNetV1 model with node_feature_dim={args.visnet_mfe_node_feature_dim}, "
              f"hidden_dims={args.visnet_mfe_hidden_dims}, output_dims={args.visnet_mfe_output_dims}, "
              f"dropout_rate={args.dropout}")
    elif args.model == 'extended':
        # 根据特征配置选择特征维度
        feature_config = get_feature_config(args.feature_config)
        
        # 设置默认特征维度为0（不存在）
        physchem_dim = 0  # 物化特征维度
        toxicity_dim = 0  # 毒性特征维度
        chromato_dim = 0  # 色谱特征维度
        
        # 根据特征配置调整维度
        if 'physchem' in feature_config:
            physchem_dim = 4  # 物化特征维度为4
        if 'toxicity' in feature_config:
            toxicity_dim = 4  # 毒性特征维度为4
        if 'chromatography' in feature_config:
            chromato_dim = 2  # 色谱特征维度为2
            
        model = ExtendedMolecularGraphNeuralNetwork(
            N, dim, layer_hidden, layer_output,
            physchem_feature_dim=physchem_dim,
            toxicity_feature_dim=toxicity_dim,
            chromato_feature_dim=chromato_dim).to(device)
        print(f"Using extended model with feature configuration: {args.feature_config}")
        print(f"Feature dimensions - PhysChem: {physchem_dim}, Toxicity: {toxicity_dim}, Chromato: {chromato_dim}")
    else:  # basic model
        model = MolecularGraphNeuralNetwork(
                N, dim, layer_hidden, layer_output, args.dropout).to(device)
        print("Using basic GNN model")
    return model

def get_model_specific_args(parser):
    """为不同模型添加特定的命令行参数"""
    # 基本模型参数组
    # basic_group = parser.add_argument_group('Basic Model Parameters')
    
    # 扩展模型参数组
    extended_group = parser.add_argument_group('Extended Model Parameters')
    extended_group.add_argument('--extended-model', action='store_true',
                                help='Use extended model with additional features (deprecated, use --model extended)')
    extended_group.add_argument('--additional-features-dim', type=int, default=None,
                                help='Dimension of additional features (default: None - auto inferred)')
    extended_group.add_argument('--feature-config', type=str, default='basic',
                                help='Feature configuration for extended model. '
                                     'Supports single config (basic, physchem, toxicity, chromatography, all) '
                                     'or comma-separated combination like "basic,toxicity" (default: basic)')
    
    # VisNet 模型参数组
    visnet_group = parser.add_argument_group('VisNet Model Parameters')
    visnet_group.add_argument('--visnet', action='store_true',
                              help='Use VisNet model (deprecated, use --model visnet)')
    visnet_group.add_argument('--visnet-hidden-channels', type=int, default=128,
                              help='Hidden channels for VisNet model')
    visnet_group.add_argument('--visnet-num-layers', type=int, default=6,
                              help='Number of layers for VisNet model')
    visnet_group.add_argument('--visnet-num-heads', type=int, default=8,
                              help='Number of heads for VisNet model')
    visnet_group.add_argument('--visnet-num-rbf', type=int, default=32,
                              help='Number of RBF functions for VisNet model')
    visnet_group.add_argument('--visnet-cutoff', type=float, default=5.0,
                              help='Cutoff distance for VisNet model')
    
    # VisNetV1 模型参数组
    visnet_v1_group = parser.add_argument_group('VisNetV1 Model Parameters')
    visnet_v1_group.add_argument('--visnet-mfe-node-feature-dim', type=int, default=64,
                                  help='Node feature dimension for VisNetV1 model')
    visnet_v1_group.add_argument('--visnet-mfe-hidden-dims', type=int, nargs='+', default=[128, 256],
                                  help='Hidden dimensions for VisNetV1 model')
    visnet_v1_group.add_argument('--visnet-mfe-output-dims', type=int, nargs='+', default=[256, 128, 1],
                                  help='Output dimensions for VisNetV1 model')
    
    # VisNetV2 模型参数组
    visnet_v2_group = parser.add_argument_group('VisNetV2 Model Parameters')
    visnet_v2_group.add_argument('--visnet-v2', action='store_true',
                                 help='Use VisNetV2 model (deprecated, use --model visnet_v2)')
    visnet_v2_group.add_argument('--visnet-v2-node-feature-dim', type=int, default=64,
                                 help='Node feature dimension for VisNetV2 model')
    visnet_v2_group.add_argument('--visnet-v2-physchem-feature-dim', type=int, default=4,
                                 help='Input dimension of physical chemistry features for VisNetV2 model (default: 4)')
    visnet_v2_group.add_argument('--visnet-v2-toxicity-feature-dim', type=int, default=4,
                                 help='Toxicity feature dimension for VisNetV2 model')
    visnet_v2_group.add_argument('--visnet-v2-chromato-feature-dim', type=int, default=2,
                                 help='Chromatography feature dimension for VisNetV2 model')
    visnet_v2_group.add_argument('--visnet-v2-graph-hidden-dim', type=int, default=512,
                                 help='Graph hidden dimension for VisNetV2 model')
    visnet_v2_group.add_argument('--visnet-v2-physchem-hidden-dim', type=int, default=128,
                                 help='Physical chemistry hidden dimension for VisNetV2 model')
    visnet_v2_group.add_argument('--visnet-v2-toxicity-hidden-dim', type=int, default=128,
                                 help='Toxicity hidden dimension for VisNetV2 model')
    visnet_v2_group.add_argument('--visnet-v2-chromato-hidden-dim', type=int, default=32,
                                 help='Chromatography hidden dimension for VisNetV2 model')
    visnet_v2_group.add_argument('--visnet-v2-fusion-hidden-dims', type=int, nargs='+', default=[512, 256, 128],
                                 help='Fusion hidden dimensions for VisNetV2 model')
    visnet_v2_group.add_argument('--visnet-v2-num-layers', type=int, default=6,
                                 help='Number of layers for VisNetV2 model')
    visnet_v2_group.add_argument('--visnet-v2-num-heads', type=int, default=8,
                                 help='Number of heads for VisNetV2 model')
    visnet_v2_group.add_argument('--visnet-v2-dropout-rate', type=float, default=0.0,
                                 help='Dropout rate for VisNetV2 model')
    visnet_v2_group.add_argument('--visnet-v2-use-attention', action='store_true',
                        help='Whether to use attention mechanism in VisNetV2')
    visnet_v2_group.add_argument('--visnet-v2-use-gating', action='store_true',
                        help='Whether to use gating mechanism in VisNetV2')
    visnet_v2_group.add_argument('--visnet-v2-feature-level', type=str, default='all',
                        choices=['graph', 'graph_physchem', 'graph_physchem_toxicity', 'all'],
                        help='Feature level for VisNetV2 model')
    visnet_v2_group.add_argument('--visnet-v2-physchem-mask', type=int, nargs='+', default=None,
                        help='Physical-chemical feature mask (1 for used, 0 for masked). '
                             'Example: "--visnet-v2-physchem-mask 1 0 1 1" to exclude the second feature (e.g. KoC)')
    visnet_v2_group.add_argument('--visnet-v2-toxicity-mask', type=int, nargs='+', default=None,
                        help='Toxicity feature mask (1 for used, 0 for masked). '
                             'Example: "--visnet-v2-toxicity-mask 1 0 1 1" to exclude the second feature')
    visnet_v2_group.add_argument('--visnet-v2-chromato-mask', type=int, nargs='+', default=None,
                        help='Chromatography feature mask (1 for used, 0 for masked). '
                             'Example: "--visnet-v2-chromato-mask 1 0" to exclude the second feature')

    # 添加dropout参数
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate for all models (default: 0.0)')
    
    return parser

if __name__ == "__main__":
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='Train Molecular Graph Neural Network')
    
    # 通用参数
    parser.add_argument('-ds', '--debug-size', type=int, default=None, 
                        help='Number of samples to use for debugging (default: None, use all data)')
    parser.add_argument('-it', '--iteration', type=int, default=150,
                        help='Number of training iterations (default: 150)')
    parser.add_argument('--batch-train', type=int, default=256,
                        help='Training batch size (default: 32)')
    parser.add_argument('--batch-test', type=int, default=128,
                        help='Test batch size (default: 32)')
    parser.add_argument('--standardize', action='store_true',
                        help='Whether to standardize the property values')
    parser.add_argument('--standardize-features', action='store_true',
                        help='Whether to standardize additional features for VisNetV2 model')
    
    # 模型选择参数
    parser.add_argument('--model', type=str, default='basic', choices=['basic', 'extended', 'visnet', 'visnet_v2', 'visnet_v1'],
                        help='Model type to use (default: basic)')
    
    # 预处理参数
    parser.add_argument('--preprocess-only', action='store_true',
                        help='Only preprocess VisNet data without training')
    
    # 优化器和调度器参数
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Initial learning rate (default: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay (L2 penalty) (default: 1e-5)')
    parser.add_argument('--scheduler-patience', type=int, default=10,
                        help='Scheduler patience (default: 10)')
    parser.add_argument('--scheduler-factor', type=float, default=0.5,
                        help='Scheduler factor (default: 0.5)')
    parser.add_argument('--min-lr', type=float, default=1e-6,
                        help='Minimum learning rate (default: 1e-6)')
    parser.add_argument('--early-stopping-patience', type=int, default=50,
                        help='Early stopping patience (default: 50)')
    
    # 损失函数参数
    parser.add_argument('--loss-function', type=str, default='mse', choices=['mse', 'l1'],
                        help='Loss function to use (default: mse)')
    
    # 内存优化参数
    parser.add_argument('--max-split-size-mb', type=int, default=None,
                        help='PyTorch max_split_size_mb to avoid memory fragmentation')
    
    # 添加保存简化模型的选项
    parser.add_argument('--save-shap-model', action='store_true',
                        help='Save a simplified model for SHAP analysis (VisNetV2 only)')
    
    # 添加模型特定参数
    parser = get_model_specific_args(parser)
    
    args = parser.parse_args()
    
    # 兼容旧参数
    if args.extended_model:
        args.model = 'extended'
    
    # 设置PyTorch内存管理参数以避免内存碎片
    if args.max_split_size_mb is not None:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f'max_split_size_mb:{args.max_split_size_mb}'
        print(f"Set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:{args.max_split_size_mb}")
    
    # 启用CUDA内存优化
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # 优化CUDA性能
        # 设置环境变量以减少内存碎片
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    radius=1
    dim=48
    layer_hidden=6
    layer_output=6
    # 使用较小的默认批处理大小以减少显存占用
    batch_train=args.batch_train
    batch_test=args.batch_test
    lr=args.learning_rate
    lr_decay=0.9
    decay_interval=200
    iteration=args.iteration
    N=5000
    # path='./data/'
    # dataname='SMRT'
    path='./data/MMF-3/'
    # INFO
    # dataname='MMF_GNN_neg'
    dataname='MMF_GNN_pos'

    np.random.seed(1234)
    torch.manual_seed(1234)

    if args.model == 'visnet' or args.model == 'visnet_v1':
        _visnet_train += dataname
        _visnet_test += dataname
    elif args.model == 'visnet_v2':
        _visnet_train += 'v2_' + dataname
        _visnet_test += 'v2_' + dataname
    
    # 创建基于时间和参数的目录
    file_paths = create_log_dir(path, dataname, dim, layer_hidden, layer_output, batch_train, lr, iteration, args.debug_size)
    log_dir = file_paths['log_dir']
    file_MAEs = file_paths['file_MAEs']
    file_test_result = file_paths['file_test_result']
    file_predictions = file_paths['file_predictions']
    file_model = file_paths['file_model']
    loss_plot_file = file_paths['loss_plot_file']
    cp_plot_file = file_paths['cp_plot_file']
    timestamp = file_paths['timestamp']  # 获取 timestamp
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print_training_setup_info(device)
    else:
        device = torch.device('cpu')
        print_training_setup_info(device)
    
    # 检查训练集和测试集文件是否存在，如果不存在则从原始文件中拆分
    train_file_path, test_file_path = split_raw_data_if_needed(path, dataname, target_column_name='Pred_RTI_Negative_ESI' if 'neg' in dataname else 'Pred_RTI_Positive_ESI')
    
    # 在加载数据集之前，加载固定字典以确保所有后续操作使用相同的词汇表
    load_fixed_dictionaries(path, dataname)
    
    # 根据是否为调试模式决定加载的数据量
    max_data = args.debug_size if args.debug_size is not None else None
    
    # 数据加载和预处理
    print(f"Loading datasets for {args.model} model...")
    
    # 根据模型类型确定是否需要额外特征
    need_additional_features = (args.model == 'extended')
    
    # 根据特征配置选择特征
    feature_config = None
    if need_additional_features:
        feature_config = get_feature_config(args.feature_config)
    
    # 加载数据集
    dataset_train, dataset_dev, dataset_test, additional_features_dim, additional_features_mean, additional_features_std = load_datasets(
        args, path, dataname, max_data)  # 移除不需要的参数

    # 标准化数据（如果需要）
    dataset_train, dataset_dev, dataset_test, property_mean, property_std = standardize_datasets_if_needed(
        args, dataset_train, dataset_dev, dataset_test, 
        additional_features_mean=additional_features_mean, 
        additional_features_std=additional_features_std)
    
    # 初始化预处理数据变量
    train_preprocessed_data, dev_preprocessed_data, test_preprocessed_data = None, None, None
    
    # 根据模型类型进行特定预处理
    if args.model in ['visnet', 'visnet_v2', 'visnet_v1']:
        # VisNet系列模型需要特殊的预处理
        print("Preprocessing VisNet-specific data...")
        preprocess_result = preprocess_visnet_data(args, dataset_train, dataset_dev, dataset_test, dataname, _visnet_train, _visnet_test)
        
        # 根据返回值数量处理结果
        if len(preprocess_result) == 7:
            # 包含标准化信息
            train_preprocessed_data, dev_preprocessed_data, test_preprocessed_data, \
            dataset_train_filtered, dataset_dev_filtered, dataset_test_filtered, standardization_info = preprocess_result
        else:
            # 不包含标准化信息
            train_preprocessed_data, dev_preprocessed_data, test_preprocessed_data, \
            dataset_train_filtered, dataset_dev_filtered, dataset_test_filtered = preprocess_result
            standardization_info = None
        
        # 根据特征级别过滤预处理数据中的特征
        if args.model == 'visnet_v2':
            # 根据特征级别过滤特征
            for data_dict in [train_preprocessed_data, dev_preprocessed_data, test_preprocessed_data]:
                for smiles in data_dict:
                    if args.visnet_v2_feature_level == 'graph':
                        # 只保留图特征，将其他特征设置为None
                        data_dict[smiles]['physchem_features'] = None
                        data_dict[smiles]['toxicity_features'] = None
                        data_dict[smiles]['chromato_features'] = None
                    elif args.visnet_v2_feature_level == 'graph_physchem':
                        # 保留图特征和物化特征，将其他特征设置为None
                        data_dict[smiles]['toxicity_features'] = None
                        data_dict[smiles]['chromato_features'] = None
                    elif args.visnet_v2_feature_level == 'graph_physchem_toxicity':
                        # 保留图特征、物化特征和毒性特征，将色谱特征设置为None
                        data_dict[smiles]['chromato_features'] = None
                    # 如果是'all'级别，则保留所有特征
        
        # 使用过滤后的数据集
        dataset_train = dataset_train_filtered
        dataset_dev = dataset_dev_filtered
        dataset_test = dataset_test_filtered
    else:
        # 基础模型和其他模型不需要额外预处理
        print(f"No additional preprocessing needed for {args.model} model")
    
    # print(dataset_test[0], property_mean, property_std)     # TEST
    
    # 使用日志工具函数打印预处理信息
    print_preprocess_info(dataset_train, dataset_dev, dataset_test)
    
    # 根据参数选择模型类型
    model = get_model(args, device, N, dim, layer_hidden, layer_output, additional_features_dim)
    
    # 创建训练器和测试器
    is_visnet_model = args.model in ['visnet', 'visnet_v2', 'visnet_v1']
    trainer = Trainer(model, visnet=is_visnet_model, batch_train=batch_train, device=device, loss_function=args.loss_function, MoleculeCacheName=_visnet_train)
    tester = Tester(model, visnet=is_visnet_model, batch_test=batch_train, device=device, name="base-tester", MoleculeCacheName=_visnet_test + "-test")
    
    # 如果使用VisNet模型，设置预处理数据
    if is_visnet_model:
        trainer.set_preprocessed_data(train_preprocessed_data)
        # 为训练集创建专用测试器train
        train_tester = Tester(model, visnet=True, batch_test=batch_test, device=device, name="train-tester", MoleculeCacheName=_visnet_train)
        train_tester.set_preprocessed_data(train_preprocessed_data)
        # 创建独立的开发集
        dev_tester = Tester(model, visnet=True, batch_test=batch_test, device=device, name="dev-tester", MoleculeCacheName=_visnet_train)
        dev_tester.set_preprocessed_data(dev_preprocessed_data)
        # 创建 Tester
        tester.set_preprocessed_data(test_preprocessed_data)
    else:
        # 对于非VisNet模型，创建开发集测试器
        dev_tester = Tester(model, visnet=False, batch_test=batch_test, device=device)
        train_tester = tester

    # 使用日志工具函数打印模型信息
    print_model_info(model)
    
    result = 'Epoch\tTime(sec)\tLoss_train\tMAE_train\tMSE_train\tR2_train\tPCC_train\tMAE_dev\tMSE_dev\tR2_dev\tPCC_dev\tMAE_test\tMSE_test\tR2_test\tPCC_test'
    with open(file_MAEs, 'w') as f:
        f.write(result + '\n')

    # 使用日志工具函数打印训练开始信息
    print_training_start_info()
    np.random.seed(1234)       
    start = timeit.default_timer()

    # 初始化最佳验证损失
    best_val_loss = float('inf')
    
    # 添加早停机制相关变量
    patience_limit = args.early_stopping_patience
    patience_counter = 0  # 计数器，记录没有改善的epoch数
    best_model_state = None
    
    # 在训练开始前设置优化器和学习率调度器
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=args.scheduler_patience, 
        factor=args.scheduler_factor, min_lr=args.min_lr)
    
    trainer.set_optimizer(optimizer)
    trainer.set_scheduler(scheduler)
    
    # 使用tqdm包装epoch循环以显示进度
    for epoch in tqdm(range(iteration), desc="Epochs"):
        epoch += 1
        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay
        model.train()
        try:
            loss_train = trainer.train(dataset_train, lr)
        except RuntimeError as e:
            raise e
        
        model.eval()

        # 使用训练集测试器进行训练集评估
        MAE_train, MSE_train, R2_train, PCC_train, predictions_train = train_tester.test_regressor(dataset_train)
        # 使用开发集测试器进行开发集评估
        MAE_dev, MSE_dev, R2_dev, PCC_dev, _ = dev_tester.test_regressor(dataset_dev)
        
        # print(MAE_train, MAE_dev)
        
        MAE_test, MSE_test, R2_test, PCC_test, _ = tester.test_regressor(dataset_test)

        time = timeit.default_timer() - start

        if epoch == 1:
            minutes = time * iteration  / 60
            hours = int(minutes / 60)
            minutes = int(minutes - 60 * hours)
            # 使用日志工具函数打印训练时间预估信息
            print_training_time_estimate(iteration, time)
            # 使用日志工具函数打印训练表头
            print_training_header(result)

        results  = '\t'.join(map(str, [epoch, time, loss_train, MAE_train, MSE_train, R2_train, PCC_train,
                                     MAE_dev, MSE_dev, R2_dev, PCC_dev, MAE_test, MSE_test, R2_test, PCC_test]))
        tester.save_MAEs(results , file_MAEs)
        
        # 早停机制逻辑
        if MAE_dev <= best_val_loss:
            best_val_loss = MAE_dev
            patience_counter = 0  # 重置计数器
            best_model_state = model.state_dict()  # 保存最佳模型状态
            tester.save_model(model, file_model)
        else:
            patience_counter += 1  # 增加计数器
            
        # 检查是否达到早停限制
        if patience_counter >= patience_limit:
            print_early_stopping_info(epoch)
            break

    # 训练结束后保存最佳模型（如果没有在早停时保存）
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        tester.save_model(model, file_model)
            
    # 保存训练参数到JSON文件
    training_params = {
        'timestamp': timestamp,
        'model_type': args.model,
        'radius': radius,
        'dim': dim,
        'layer_hidden': layer_hidden,
        'layer_output': layer_output,
        'batch_train': batch_train,
        'batch_test': batch_test,
        'learning_rate': lr,
        'lr_decay': lr_decay,
        'decay_interval': decay_interval,
        'iteration': iteration,
        'N': N,
        'path': path,
        'dataname': dataname,
        'debug_size': args.debug_size,
        'standardize': args.standardize,
        'standardize_features': args.standardize_features,  # 添加特征标准化参数
        'property_mean': float(property_mean) if property_mean is not None else None,
        'property_std': float(property_std) if property_std is not None else None,
        'additional_features_mean': additional_features_mean.tolist() if additional_features_mean is not None and hasattr(additional_features_mean, 'tolist') else additional_features_mean,
        'additional_features_std': additional_features_std.tolist() if additional_features_std is not None and hasattr(additional_features_std, 'tolist') else additional_features_std,
        'feature_config': feature_config,
        'additional_features_dim': additional_features_dim,
        'early_stopping_patience': args.early_stopping_patience,
        'scheduler_patience': args.scheduler_patience,
        'scheduler_factor': args.scheduler_factor,
        'min_lr': args.min_lr,
        'weight_decay': args.weight_decay,
        'loss_function': args.loss_function,
        'dropout': args.dropout
    }
        
    # 为VisNet模型添加特定参数
    if args.model == 'visnet':
        training_params.update({
            'visnet_hidden_channels': args.visnet_hidden_channels,
            'visnet_num_layers': args.visnet_num_layers,
            'visnet_num_heads': args.visnet_num_heads,
            'visnet_num_rbf': args.visnet_num_rbf,
            'visnet_cutoff': args.visnet_cutoff
        })
    elif args.model == 'visnet_v1':
        training_params.update({
            'visnet_mfe_node_feature_dim': args.visnet_mfe_node_feature_dim,
            'visnet_mfe_hidden_dims': args.visnet_mfe_hidden_dims,
            'visnet_mfe_output_dims': args.visnet_mfe_output_dims
        })
    elif args.model == 'visnet_v2':
        training_params.update({
            'visnet_v2_node_feature_dim': args.visnet_v2_node_feature_dim,
            'visnet_v2_physchem_feature_dim': args.visnet_v2_physchem_feature_dim,
            'visnet_v2_toxicity_feature_dim': args.visnet_v2_toxicity_feature_dim,
            'visnet_v2_chromato_feature_dim': args.visnet_v2_chromato_feature_dim,
            'visnet_v2_graph_hidden_dim': args.visnet_v2_graph_hidden_dim,
            'visnet_v2_physchem_hidden_dim': args.visnet_v2_physchem_hidden_dim,
            'visnet_v2_toxicity_hidden_dim': args.visnet_v2_toxicity_hidden_dim,
            'visnet_v2_chromato_hidden_dim': args.visnet_v2_chromato_hidden_dim,
            'visnet_v2_fusion_hidden_dims': args.visnet_v2_fusion_hidden_dims,
            'visnet_v2_use_attention': args.visnet_v2_use_attention,
            'visnet_v2_use_gating': args.visnet_v2_use_gating,
            'visnet_v2_feature_level': args.visnet_v2_feature_level,
            'visnet_v2_physchem_mask': args.visnet_v2_physchem_mask,
            'visnet_v2_toxicity_mask': args.visnet_v2_toxicity_mask,
            'visnet_v2_chromato_mask': args.visnet_v2_chromato_mask,
            'standardization_info': standardization_info  # 添加标准化信息
        })
        
    save_training_params(training_params, log_dir)
    
    # 绘制训练指标图
    try:
        plot_training_metrics(file_MAEs, loss_plot_file, 
                            json_file=os.path.join(log_dir, 'training_metrics.json'))
    except Exception as e:
        print(f"Warning: Could not plot training metrics: {e}")
    
    # 测试最佳模型
    model.eval()
    
    # 确保BatchNorm层处于评估状态
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            module.eval()
    
    MAE_train, MSE_train, R2_train, PCC_train, predictions_train = train_tester.test_regressor(dataset_train)
    MAE_dev, MSE_dev, R2_dev, PCC_dev, predictions_dev = dev_tester.test_regressor(dataset_dev)
    MAE_test, MSE_test, R2_test, PCC_test, predictions_test = tester.test_regressor(dataset_test)
    
    # 添加调试信息
    print("=== Training Final Test Results ===")
    print(f"Train MAE: {MAE_train:.6f}, RMSE: {np.sqrt(MSE_train):.6f}, R2: {R2_train:.6f}")
    print(f"Dev MAE: {MAE_dev:.6f}, RMSE: {np.sqrt(MSE_dev):.6f}, R2: {R2_dev:.6f}")
    print(f"Test MAE: {MAE_test:.6f}, RMSE: {np.sqrt(MSE_test):.6f}, R2: {R2_test:.6f}")
    
    # 打印模型的部分参数作为调试信息
    print("=== Model Parameters (first 5) ===")
    for name, param in list(model.named_parameters())[:5]:
        print(f"{name}: {param.data.flatten()[:5]}...")
    
    # 打印测试数据集的一些样本信息
    print("=== Dataset Sample Info ===")
    if len(dataset_test) > 0:
        sample = dataset_test[0]
        if isinstance(sample, dict):
            print(f"Sample keys: {list(sample.keys())}")
            if 'smiles' in sample:
                print(f"First sample SMILES: {sample['smiles']}")
            if 'property_value' in sample:
                print(f"First sample property value: {sample['property_value']}")
        else:
            print(f"Sample type: {type(sample)}, length: {len(sample) if hasattr(sample, '__len__') else 'N/A'}")
    
    # 打印预处理数据的一些信息
    print("=== Preprocessed Data Info ===")
    if test_preprocessed_data:
        print(f"Number of preprocessed samples: {len(test_preprocessed_data)}")
        first_key = list(test_preprocessed_data.keys())[0]
        first_sample = test_preprocessed_data[first_key]
        print(f"First sample keys: {list(first_sample.keys())}")
        if 'physchem_features' in first_sample:
            print(f"Physchem features shape: {first_sample['physchem_features'].shape if hasattr(first_sample['physchem_features'], 'shape') else 'N/A'}")
            print(f"Physchem features sample: {first_sample['physchem_features'][:5] if hasattr(first_sample['physchem_features'], '__getitem__') else 'N/A'}")
    
torch.save(test_preprocessed_data, "test_preprocessed_data.pt")
    torch.save(dataset_test, "dataset_test.pt")
    
    # 保存最终测试结果
    with open(file_test_result, 'w') as f:
        f.write(f'MAE_train: {MAE_train}\n')
        f.write(f'MSE_train: {MSE_train}\n')
        f.write(f'R2_train: {R2_train}\n')
        f.write(f'PCC_train: {PCC_train}\n')
        f.write(f'MAE_dev: {MAE_dev}\n')
        f.write(f'MSE_dev: {MSE_dev}\n')
        f.write(f'R2_dev: {R2_dev}\n')
        f.write(f'PCC_dev: {PCC_dev}\n')
        f.write(f'MAE_test: {MAE_test}\n')
        f.write(f'MSE_test: {MSE_test}\n')
        f.write(f'R2_test: {R2_test}\n')
        f.write(f'PCC_test: {PCC_test}\n')
    
    # 保存预测结果
    tester.save_predictions(predictions_train, file_predictions.replace('.txt', '_train.txt'))
    tester.save_predictions(predictions_dev, file_predictions.replace('.txt', '_dev.txt'))
    tester.save_predictions(predictions_test, file_predictions.replace('.txt', '_test.txt'))
    
    # 绘制预测值vs实际值图
    try:
        plot_predicted_vs_actual(file_predictions.replace('.txt', '_train.txt'), 
                               os.path.join(log_dir, 'predicted_vs_actual_train.png'), 
                               property_mean, property_std,
                               json_file=os.path.join(log_dir, 'predicted_vs_actual_train.json'))
        plot_predicted_vs_actual(file_predictions.replace('.txt', '_dev.txt'), 
                               os.path.join(log_dir, 'predicted_vs_actual_dev.png'), 
                               property_mean, property_std,
                               json_file=os.path.join(log_dir, 'predicted_vs_actual_dev.json'))
        plot_predicted_vs_actual(file_predictions.replace('.txt', '_test.txt'), 
                               os.path.join(log_dir, 'predicted_vs_actual_test.png'), 
                               property_mean, property_std,
                               json_file=os.path.join(log_dir, 'predicted_vs_actual_test.json'))
    except Exception as e:
        print(f"Warning: Could not plot predicted vs actual: {e}")
    
    print(f"Training completed. Results saved to {log_dir}")