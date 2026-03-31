根据您提供的代码，我来绘制VisNetV2模型的Mermaid框架图：

```mermaid
graph TD
    A[输入数据] --> B[ViSNet 图结构特征提取]
    A --> C[物理化学特征]
    A --> D[毒性特征]
    A --> E[色谱质谱特征]
    
    B --> F[图特征处理器<br/>Linear-BN-ReLU-Dropout<br/>64→256→512]
    C --> G[物理化学特征处理器<br/>Linear-BN-ReLU-Dropout<br/>4→32→64]
    D --> H[毒性特征处理器<br/>Linear-BN-ReLU<br/>4→32→64]
    E --> I[色谱质谱特征处理器<br/>Linear-ReLU<br/>2→16→32]
    
    F --> J[多模态特征融合]
    G --> J
    H --> J
    I --> J
    
    J --> K{是否使用注意力机制?}
    K -->|是| L[MultiModalAttention<br/>多头注意力机制]
    K -->|否| M[直接特征拼接]
    
    L --> N{是否使用门控机制?}
    M --> N
    
    N -->|是| O[ModalGating<br/>门控融合机制]
    N -->|否| P[特征拼接]
    
    O --> Q[动态构建融合网络]
    P --> Q
    
    Q --> R[特征融合回归网络<br/>Linear-BN-ReLU-Dropout<br/>输入维度动态→512→256→128→1]
    
    R --> S[RT预测输出]
    
    %% 样式定义 - 黑色文字
    classDef input fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#000000
    classDef processor fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,color:#000000
    classDef attention fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000000
    classDef fusion fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px,color:#000000
    classDef output fill:#ffebee,stroke:#b71c1c,stroke-width:2px,color:#000000
    classDef decision fill:#fff9c4,stroke:#f57f17,stroke-width:2px,color:#000000
    
    class A,C,D,E input
    class F,G,H,I processor
    class L,O attention
    class J,Q,R fusion
    class S output
    class K,N decision
```

## 模型组件详细说明：

### 1. 输入模块
- **图结构数据** (z, pos, batch)
- **物理化学特征** (4维)
- **毒性特征** (4维) 
- **色谱质谱特征** (2维)

### 2. 特征处理模块
- **ViSNet**: 图神经网络，提取分子3D结构特征
- **四个特征处理器**: 分别处理不同类型的特征，包含BN、ReLU、Dropout等

### 3. 融合机制模块
- **MultiModalAttention**: 可选的多头注意力机制，动态调节各模态重要性
- **ModalGating**: 可选的门控机制，学习特征融合权重
- **动态融合网络**: 根据实际提供的特征动态构建网络结构

### 4. 输出模块
- **回归预测**: 输出单个RT值，用于保留时间预测

这个框架展示了VisNetV2的多模态、模块化设计特点，支持灵活的特征组合和可选的注意力/门控机制。