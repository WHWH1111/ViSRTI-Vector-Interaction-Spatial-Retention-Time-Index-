# VisNetV2 多特征融合与优化策略说明

## 1. 多模态特征融合路径

VisNetV2采用模块化设计，支持多种类型特征的融合。以下是完整的特征处理流程：

```mermaid
graph TD
    A[输入数据] --> B{特征预处理}
    B --> C[图结构特征<br/>VisNet Core]
    B --> D[物理化学特征<br/>PhysChem Features]
    B --> E[毒性特征<br/>Toxicity Features]
    B --> F[色谱特征<br/>Chromatographic Features]
    
    C --> G{特征融合模块}
    D --> G
    E --> G
    F --> G
    
    G --> H[融合网络<br/>Fusion Network]
    H --> I[RT预测值]
    
    style C fill:#FFE4B5,stroke:#333,stroke-width:2px,color:#000
    style D fill:#98FB98,stroke:#333,stroke-width:2px,color:#000
    style E fill:#98FB98,stroke:#333,stroke-width:2px,color:#000
    style F fill:#98FB98,stroke:#333,stroke-width:2px,color:#000
    style G fill:#87CEEB,stroke:#333,stroke-width:2px,color:#000
    style H fill:#FFA07A,stroke:#333,stroke-width:2px,color:#000
```

## 2. 各模态特征处理详情

### 2.1 图结构特征 (Graph Features)

图结构特征是模型的核心，通过VisNet核心模块提取分子的3D几何信息：

```mermaid
graph LR
    A[原子序数 z] --> B[VisNet核心模块]
    C[原子坐标 pos] --> B
    D[批次索引 batch] --> B
    B --> E[图级表示<br/>graph_features]
    
    style B fill:#FFE4B5,stroke:#333,stroke-width:2px,color:#000
    style E fill:#98FB98,stroke:#333,stroke-width:2px,color:#000
```

### 2.2 物理化学特征 (PhysChem Features)

物理化学特征包括分子层面的重要属性：

```mermaid
graph LR
    A[物理化学特征输入<br/>4维] --> B{特征掩码}
    B -- 掩码后特征 --> C[特征处理器<br/>physchem_processor]
    C --> D[处理后特征<br/>physchem_features]
    
    style B fill:#87CEEB,stroke:#333,stroke-width:2px,color:#000
    style C fill:#98FB98,stroke:#333,stroke-width:2px,color:#000
    style D fill:#FFA07A,stroke:#333,stroke-width:2px,color:#000
```

### 2.3 毒性特征 (Toxicity Features)

毒性特征反映化合物的潜在毒性属性：

```mermaid
graph LR
    A[毒性特征输入<br/>4维] --> B{特征掩码}
    B -- 掩码后特征 --> C[特征处理器<br/>toxicity_processor]
    C --> D[处理后特征<br/>toxicity_features]
    
    style B fill:#87CEEB,stroke:#333,stroke-width:2px,color:#000
    style C fill:#98FB98,stroke:#333,stroke-width:2px,color:#000
    style D fill:#FFA07A,stroke:#333,stroke-width:2px,color:#000
```

### 2.4 色谱特征 (Chromatographic Features)

色谱特征来源于色谱实验数据：

```mermaid
graph LR
    A[色谱特征输入<br/>2维] --> B{特征掩码}
    B -- 掩码后特征 --> C[特征处理器<br/>chromato_processor]
    C --> D[处理后特征<br/>chromato_features]
    
    style B fill:#87CEEB,stroke:#333,stroke-width:2px,color:#000
    style C fill:#98FB98,stroke:#333,stroke-width:2px,color:#000
    style D fill:#FFA07A,stroke:#333,stroke-width:2px,color:#000
```

## 3. SHAP特征掩码策略

通过SHAP分析可以评估各特征的重要性，进而通过掩码机制选择重要特征：

```mermaid
graph TD
    A[SHAP分析] --> B{特征重要性排序}
    B --> C[选择Top-K特征]
    C --> D[生成特征掩码]
    D --> E[应用到模型]
    
    E --> F[物化特征掩码<br/>physchem_feature_mask]
    E --> G[毒性特征掩码<br/>toxicity_feature_mask]
    E --> H[色谱特征掩码<br/>chromato_feature_mask]
    
    style A fill:#FFE4B5,stroke:#333,stroke-width:2px,color:#000
    style B fill:#87CEEB,stroke:#333,stroke-width:2px,color:#000
    style C fill:#87CEEB,stroke:#333,stroke-width:2px,color:#000
    style D fill:#87CEEB,stroke:#333,stroke-width:2px,color:#000
    style E fill:#98FB98,stroke:#333,stroke-width:2px,color:#000
```

## 4. 模块维度调整策略

根据不同特征的重要性，可以调整各个处理模块的隐藏层维度：

```mermaid
graph TD
    A[模型配置] --> B{维度设置}
    
    B --> C[图特征维度<br/>graph_hidden_dim=512]
    B --> D[物化特征维度<br/>physchem_hidden_dim=64]
    B --> E[毒性特征维度<br/>toxicity_hidden_dim=64]
    B --> F[色谱特征维度<br/>chromato_hidden_dim=32]
    B --> G[融合网络维度<br/>fusion=[512,256,128]]
    
    style B fill:#87CEEB,stroke:#333,stroke-width:2px,color:#000
    style C fill:#98FB98,stroke:#333,stroke-width:2px,color:#000
    style D fill:#98FB98,stroke:#333,stroke-width:2px,color:#000
    style E fill:#98FB98,stroke:#333,stroke-width:2px,color:#000
    style F fill:#98FB98,stroke:#333,stroke-width:2px,color:#000
    style G fill:#FFA07A,stroke:#333,stroke-width:2px,color:#000
```

## 5. 完整前向传播流程

```mermaid
sequenceDiagram
    participant Input as 输入数据
    participant Graph as 图特征模块
    participant PhysChem as 物化特征模块
    participant Toxicity as 毒性特征模块
    participant Chromato as 色谱特征模块
    participant Attention as 注意力机制
    participant Gating as 门控机制
    participant Fusion as 融合网络
    participant Output as 输出层
    
    Input->>Graph: z, pos, batch
    Input->>PhysChem: physchem_features
    Input->>Toxicity: toxicity_features
    Input->>Chromato: chromato_features
    
    Graph->>Attention: graph_features
    PhysChem->>Attention: physchem_features
    Toxicity->>Attention: toxicity_features
    Chromato->>Attention: chromato_features
    
    Attention->>Gating: 加权特征
    Gating->>Fusion: 融合特征
    
    Fusion->>Output: combined_features
    Output->>Output: RT预测值
```

## 6. 特征级别控制

模型支持不同级别的特征组合：

```mermaid
graph TD
    A[特征级别控制<br/>feature_level] --> B{模式选择}
    
    B --> C[graph<br/>仅图特征]
    B --> D[graph_physchem<br/>图+物化特征]
    B --> E[graph_physchem_toxicity<br/>图+物化+毒性特征]
    B --> F[all<br/>全部特征]
    
    style A fill:#87CEEB,stroke:#333,stroke-width:2px,color:#000
    style B fill:#87CEEB,stroke:#333,stroke-width:2px,color:#000
```