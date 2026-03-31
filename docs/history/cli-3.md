# Train

## test
```bash
CUDA_VISIBLE_DEVICES=1 python train.py -it 50 --batch-train 16 --standardize
CUDA_VISIBLE_DEVICES=1 python train.py -it 50 -ds 200 --standardize --loss-function l1 --model visnet_v1

```

## real
```bash
CUDA_VISIBLE_DEVICES=1 python train.py --standardize --loss-function l1
CUDA_VISIBLE_DEVICES=1 python train.py --standardize --loss-function l1 --model visnet_v1 --batch-train 64
```

## visnet-v2
```bash
# 仅使用图特征（基础指纹）
CUDA_VISIBLE_DEVICES=2 python train.py --model visnet_v2 --visnet-v2-feature-level graph --standardize --loss-function l1 --batch-train 64 -it 10 --standardize-features

# 使用图特征 + 物化特征（黄色背景）
CUDA_VISIBLE_DEVICES=2 python train.py --model visnet_v2 --visnet-v2-feature-level graph_physchem --standardize --loss-function l1 --batch-train 64 --standardize-features -it 200

# 使用图特征 + 物化特征 + 毒性特征（粉色背景）
CUDA_VISIBLE_DEVICES=0 python train.py --model visnet_v2 --visnet-v2-feature-level graph_physchem_toxicity --standardize --loss-function l1 --batch-train 64 --batch-test 64 --standardize-features \
 -it 5 -ds 200

# 使用所有特征（绿色背景）
CUDA_VISIBLE_DEVICES=2 python train.py --model visnet_v2 --visnet-v2-feature-level all --standardize --loss-function l1 --batch-train 64 --batch-test 64 --standardize-features \
 -it 5 -ds 200
```

## extend-model
```bash
# 训练基础指纹模型（黄色背景数据的第一步）
python train.py --model extended --feature-config basic 

# 训练基础指纹 + 物化特征模型（黄色背景数据）
CUDA_VISIBLE_DEVICES=1 python train.py --model extended --feature-config basic,physchem -it 5 -ds 200

# 训练基础指纹 + 物化特征 + 毒性特征模型（粉色背景数据）
python train.py --model extended --feature-config basic,physchem,toxicity 

# 训练使用所有特征的模型（绿色背景数据）
python train.py --model extended --feature-config all --standardize --loss-function l1
```

## Mask

```bash
# 排除物化特征中的Koc_predicted特征（索引为3）
python train.py --model visnet_v2 --visnet-v2-physchem-mask 1 1 1 0

CUDA_VISIBLE_DEVICES=4 python train.py --model visnet_v2 --visnet-v2-feature-level graph_physchem_toxicity --standardize --loss-function l1 --batch-train 64 --standardize-features --visnet-v2-physchem-mask 1 1 1 0

# 只使用Monoiso_Mass和Koc_predicted特征（索引为0和3）
python train.py --model visnet_v2 --visnet-v2-physchem-mask 1 0 0 1

# 排除物化特征中的logKow/Exp_logKow特征（索引为1）
python train.py --model visnet_v2 --visnet-v2-physchem-mask 1 0 1 1
```

### 物化特征索引说明

物化特征共有4个，其索引位置如下：

- 索引 0: Monoiso_Mass
- 索引 1: logKow/Exp_logKow (优先使用Exp_logKow_EPISuite，否则使用logKow_EPISuite)
- 索引 2: alogp/xlogp (优先使用xlogp_ChemSpider，否则使用alogp_ChemSpider)
- 索引 3: Koc_predicted

### 使用说明

特征掩码是一个由0和1组成的序列，其中：
- 1 表示使用该特征
- 0 表示排除该特征

掩码的长度应与物化特征的维度一致（当前为4）。

### tasks

```bash
# pos
CUDA_VISIBLE_DEVICES=2 python train.py --model visnet_v2 --visnet-v2-feature-level graph_physchem_toxicity --standardize --loss-function l1 --batch-train 256 --batch-test 2 --standardize-features --visnet-v2-physchem-mask 1 1 1 0  --it 1

python train.py --model visnet_v2 --visnet-v2-feature-level graph_physchem_toxicity --standardize --loss-function l1 --batch-train 64 --standardize-features --visnet-v2-physchem-mask 1 1 1 0  --visnet-v2-toxicity-mask 1 0 1 1

# neg
CUDA_VISIBLE_DEVICES=0 
python train.py --model visnet_v2 --visnet-v2-feature-level all --standardize --loss-function l1 --batch-train 64 --batch-test 64 --standardize-features --visnet-v2-physchem-mask 1 0 1 1 -it 2
```


# Task-3

```bash
```