import torch
from torch import nn
from .visnet_core import ViSNet
import torch.nn.functional as F


class VisNetV1(nn.Module):
    """
    VisNet分子特征提取器

    该模型将SMILES转换为3D几何结构后，通过VisNet提取64维graph_rep，
    再通过两层全连接层(64->128->256)扩展维度，
    最后通过全局池化生成512维分子特征表示，
    再通过几层线性层将维度降低到1维输出。
    """
    
    def __init__(self, node_feature_dim: int = 64, hidden_dims: list = None, output_dims: list = None, 
                 dropout_rate: float = 0.5):
        """
        初始化特征提取器

        Args:
            node_feature_dim: VisNet输出的graph_rep维度 (64)
            hidden_dims: 全连接层维度配置 [128, 256]
            output_dims: 输出层维度配置，默认为[256, 128, 1]
            dropout_rate: dropout比率，默认为0.5
        """
        super(VisNetV1, self).__init__()
        
        # VisNet基础模型，用于提取64维graph_rep
        self.visnet = ViSNet(
            hidden_channels=128,  # 确保输出64维graph_rep
            num_layers=6,
            num_rbf=32,
            cutoff=5.0
        )
        
        self.node_feature_dim = node_feature_dim
        self.hidden_dims = hidden_dims if hidden_dims is not None else [128, 256]
        self.dropout_rate = dropout_rate
        
        # 全连接层：64 -> 128 -> 256
        self.fc_layers = nn.ModuleList()
        self.fc_dropout_layers = nn.ModuleList()  # 添加dropout层列表
        input_dim = node_feature_dim
        for hidden_dim in self.hidden_dims:
            self.fc_layers.append(nn.Linear(input_dim, hidden_dim))
            self.fc_dropout_layers.append(nn.Dropout(self.dropout_rate))  # 为每层添加dropout
            input_dim = hidden_dim
            
        # 输出层：512 -> 256 -> 128 -> 1
        self.output_dims = output_dims if output_dims is not None else [256, 128, 1]
        self.output_layers = nn.ModuleList()
        self.output_dropout_layers = nn.ModuleList()  # 添加输出dropout层列表
        # 输入维度是fc_layers的最后一层输出维度的2倍（因为拼接了mean和max）
        input_dim = self.hidden_dims[-1] * 2
        for output_dim in self.output_dims:
            self.output_layers.append(nn.Linear(input_dim, output_dim))
            self.output_dropout_layers.append(nn.Dropout(self.dropout_rate))  # 为每层添加dropout
            input_dim = output_dim
    
    def forward(self, data) -> torch.Tensor:
        """
        前向传播，提取分子特征

        Args:
            data: 包含原子序数(z)、坐标(pos)和批次信息(batch)的图数据

        Returns:
            分子特征表示 (1维输出)
        """
        z, pos, batch = data.z, data.pos, data.batch
        
        # 通过VisNet的完整forward方法获取64维graph_rep
        _, graph_rep = self.visnet(z, pos, batch)
        
        # 通过全连接层扩展维度: 64 -> 128 -> 256
        expanded_features = graph_rep
        for i, fc_layer in enumerate(self.fc_layers):
            expanded_features = fc_layer(expanded_features)
            if i < len(self.fc_layers) - 1:  # 最后一层不加激活函数
                expanded_features = F.relu(expanded_features)
                expanded_features = self.fc_dropout_layers[i](expanded_features)  # 应用dropout
        
        # 重塑特征以匹配池化操作的期望输入
        # 注意：graph_rep已经是图级别的表示，我们将其复制以模拟节点级特征用于池化
        # batch_size = expanded_features.size(0)
        
        # 直接将扩展后的图级别特征作为最终特征（每个图一个512维向量）
        # 由于graph_rep已经是图级别的表示，我们可以将其直接复制为mean和max部分
        x_mean = expanded_features
        x_max = expanded_features
        
        # 合并mean和max池化结果
        x = torch.cat([x_mean, x_max], dim=1)
        
        # 通过输出层将维度降低到1维
        for i, (output_layer, dropout_layer) in enumerate(zip(self.output_layers, self.output_dropout_layers)):
            x = output_layer(x)
            if i < len(self.output_layers) - 1:  # 最后一层不加激活函数
                x = F.relu(x)
                x = dropout_layer(x)  # 应用dropout
        
        return x