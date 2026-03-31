import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ExtendedMolecularGraphNeuralNetwork(nn.Module):
    def __init__(self, N, dim, layer_hidden, layer_output, 
                 physchem_feature_dim: int = 4,     # 物化特征维度
                 toxicity_feature_dim: int = 4,     # 毒性特征维度
                 chromato_feature_dim: int = 2):    # 色谱特征维度
        super(ExtendedMolecularGraphNeuralNetwork, self).__init__()
        # 原有的GNN组件
        self.embed_fingerprint = nn.Embedding(N, dim)
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim)
                                            for _ in range(layer_hidden)])
        
        # 保存各特征维度
        self.physchem_feature_dim = physchem_feature_dim
        self.toxicity_feature_dim = toxicity_feature_dim
        self.chromato_feature_dim = chromato_feature_dim
        
        # 1. 分子图特征处理器
        self.molecule_processor = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # 2. 物理化学特征处理器
        if self.physchem_feature_dim > 0:
            self.physchem_processor = nn.Sequential(
                nn.Linear(self.physchem_feature_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU()
            )
        else:
            self.physchem_processor = None
        
        # 3. 毒性特征处理器
        if self.toxicity_feature_dim > 0:
            self.toxicity_processor = nn.Sequential(
                nn.Linear(self.toxicity_feature_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU()
            )
        else:
            self.toxicity_processor = None
        
        # 4. 色谱特征处理器
        if self.chromato_feature_dim > 0:
            self.chromato_processor = nn.Sequential(
                nn.Linear(self.chromato_feature_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 32),
                nn.ReLU()
            )
        else:
            self.chromato_processor = None
        
        # 5. 融合网络 - 将在forward中动态构建
        self.fusion_net = None
        self._last_feature_dim = None
        
        # 输出层
        self.output_layer = nn.Linear(128, 1)  # 最终输出RT值
        
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def _build_fusion_net(self, input_dim):
        """动态构建融合网络"""
        self.fusion_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

    def pad(self, matrices, pad_value):
        """Pad the list of matrices
        with a pad_value (e.g., 0) for batch processing.
        For example, given a list of matrices [A, B, C],
        we obtain a new matrix [A00, 0B0, 00C],
        where 0 is the zero (i.e., pad value) matrix.
        """
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        device = matrices[0].device if matrices else torch.device('cpu')
        zeros = torch.FloatTensor(np.zeros((M, N))).to(device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i:i+m, j:j+n] = matrix
            i += m
            j += n
        return pad_matrices

    def update(self, matrix, vectors, layer):
        hidden_vectors = torch.relu(self.W_fingerprint[layer](vectors))
        return hidden_vectors + torch.matmul(matrix, hidden_vectors)

    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)

    def mean(self, vectors, axis):
        mean_vectors = [torch.mean(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(mean_vectors)

    def gnn(self, inputs):
        """Cat or pad each input data for batch processing."""
        # 根据输入数据的实际格式进行解包
        if len(inputs) >= 4:
            # 标准格式：(Smiles, fingerprints, adjacencies, molecular_sizes, ...)
            Smiles, fingerprints, adjacencies, molecular_sizes = inputs[0:4]
        else:
            # 简化格式：(fingerprints, adjacencies, molecular_sizes)
            Smiles = None
            fingerprints, adjacencies, molecular_sizes = inputs[0:3]
            
        fingerprints = torch.cat(fingerprints)
        adjacencies = self.pad(adjacencies, 0)

        """GNN layer (update the fingerprint vectors)."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        for layer_index in range(len(self.W_fingerprint)):
            hs = self.update(adjacencies, fingerprint_vectors, layer_index)
            fingerprint_vectors = F.normalize(hs, 2, 1)  # normalize.

        """Molecular vector by sum or mean of the fingerprint vectors."""
        molecular_vectors = self.sum(fingerprint_vectors, molecular_sizes)

        return Smiles, molecular_vectors

    def forward_regressor(self, data_batch, train):
        # 解析输入数据
        if len(data_batch) == 8:  
            # 新格式: smiles, fingerprints, adjacencies, molecular_sizes, 
            #        physchem_features, toxicity_features, chromato_features, correct_values
            inputs = data_batch[:4]  # smiles, fingerprints, adjacencies, molecular_sizes
            physchem_features = data_batch[4]   # 物化特征
            toxicity_features = data_batch[5]   # 毒性特征
            chromato_features = data_batch[6]   # 色谱特征
            correct_values = data_batch[7]      # 正确值
        elif len(data_batch) == 6:
            # 中间格式: smiles, fingerprints, adjacencies, molecular_sizes, 
            #          physchem_features, correct_values
            inputs = data_batch[:4]
            physchem_features = data_batch[4]
            toxicity_features = None
            chromato_features = None
            correct_values = data_batch[5]
        elif len(data_batch) == 5:
            # 基础格式: smiles, fingerprints, adjacencies, molecular_sizes, correct_values
            inputs = data_batch[:4]
            physchem_features = None
            toxicity_features = None
            chromato_features = None
            correct_values = data_batch[4]
        else:
            raise ValueError(f"Unexpected data format with {len(data_batch)} elements")
        
        print(physchem_features[0])
        print(toxicity_features[0])
        print(chromato_features[0])
        exit(0)
        
        correct_values = torch.cat(correct_values)
        
        if train:
            Smiles, molecular_vectors = self.gnn(inputs)
            
            # 获取设备信息
            device = molecular_vectors.device
            
            # 1. 处理分子图特征
            molecule_features = self.molecule_processor(molecular_vectors)
            features_list = [molecule_features]
            feature_dims = [molecule_features.size(1)]
            
            # 2. 处理物理化学特征（如果有）
            if physchem_features is not None and self.physchem_processor is not None:
                physchem_features = torch.stack(physchem_features).to(device)
                physchem_processed = self.physchem_processor(physchem_features)
                features_list.append(physchem_processed)
                feature_dims.append(physchem_processed.size(1))
            
            # 3. 处理毒性特征（如果有）
            if toxicity_features is not None and self.toxicity_processor is not None:
                toxicity_features = torch.stack(toxicity_features).to(device)
                toxicity_processed = self.toxicity_processor(toxicity_features)
                features_list.append(toxicity_processed)
                feature_dims.append(toxicity_processed.size(1))
            
            # 4. 处理色谱特征（如果有）
            if chromato_features is not None and self.chromato_processor is not None:
                chromato_features = torch.stack(chromato_features).to(device)
                chromato_processed = self.chromato_processor(chromato_features)
                features_list.append(chromato_processed)
                feature_dims.append(chromato_processed.size(1))
            
            # 5. 特征融合
            combined_features = torch.cat(features_list, dim=1)
            
            # 6. 动态构建融合网络
            total_feature_dim = sum(feature_dims)
            if self.fusion_net is None or self._last_feature_dim != total_feature_dim:
                self._build_fusion_net(total_feature_dim)
                self._last_feature_dim = total_feature_dim
                self.fusion_net = self.fusion_net.to(device)
            
            # 7. 融合特征处理和预测
            fused_features = self.fusion_net(combined_features)
            predicted_values = self.output_layer(fused_features)
            
            # 确保correct_values的维度与predicted_values匹配
            if correct_values.dim() == 1:
                correct_values = correct_values.unsqueeze(1)
            
            loss = F.mse_loss(predicted_values, correct_values)
            return loss
        else:
            with torch.no_grad():
                Smiles, molecular_vectors = self.gnn(inputs)
                
                # 获取设备信息
                device = molecular_vectors.device
                
                # 1. 处理分子图特征
                molecule_features = self.molecule_processor(molecular_vectors)
                features_list = [molecule_features]
                feature_dims = [molecule_features.size(1)]
                
                # 2. 处理物理化学特征（如果有）
                if physchem_features is not None and self.physchem_processor is not None:
                    physchem_features = torch.stack(physchem_features).to(device)
                    physchem_processed = self.physchem_processor(physchem_features)
                    features_list.append(physchem_processed)
                    feature_dims.append(physchem_processed.size(1))
                
                # 3. 处理毒性特征（如果有）
                if toxicity_features is not None and self.toxicity_processor is not None:
                    toxicity_features = torch.stack(toxicity_features).to(device)
                    toxicity_processed = self.toxicity_processor(toxicity_features)
                    features_list.append(toxicity_processed)
                    feature_dims.append(toxicity_processed.size(1))
                
                # 4. 处理色谱特征（如果有）
                if chromato_features is not None and self.chromato_processor is not None:
                    chromato_features = torch.stack(chromato_features).to(device)
                    chromato_processed = self.chromato_processor(chromato_features)
                    features_list.append(chromato_processed)
                    feature_dims.append(chromato_processed.size(1))
                
                # 5. 特征融合
                combined_features = torch.cat(features_list, dim=1)
                
                # 6. 动态构建融合网络
                total_feature_dim = sum(feature_dims)
                if self.fusion_net is None or self._last_feature_dim != total_feature_dim:
                    self._build_fusion_net(total_feature_dim)
                    self._last_feature_dim = total_feature_dim
                    self.fusion_net = self.fusion_net.to(device)
                
                # 7. 融合特征处理和预测
                fused_features = self.fusion_net(combined_features)
                predicted_values = self.output_layer(fused_features)
                
            predicted_values = predicted_values.to('cpu').data.numpy()
            correct_values = correct_values.to('cpu').data.numpy()
            # 修复：直接返回，避免不必要的拼接操作
            return Smiles, predicted_values.flatten(), correct_values.flatten()