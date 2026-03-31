# Train

```bash
# TEST
CUDA_VISIBLE_DEVICES=0 python train.py -it 50 -ds 200 --standardize
CUDA_VISIBLE_DEVICES=0 python train.py --standardize --loss-function l1 -ds 2000 -it 50  --model visnet-v1

# TRAIN
CUDA_VISIBLE_DEVICES=6 python train.py --extended-model --iteration 200 --standardize
CUDA_VISIBLE_DEVICES=1 python train.py --extended-model --iteration 200 

# 10.22
CUDA_VISIBLE_DEVICES=0 python train.py --visnet --standardize
CUDA_VISIBLE_DEVICES=0 python train.py --visnet --standardize --batch-train 128
```

## visnet-v2

```bash
CUDA_VISIBLE_DEVICES=1 python train.py --model visnet_v2 --visnet-v2-node-feature-dim 64 --visnet-v2-physchem-feature-dim 12 --visnet-v2-toxicity-feature-dim 4 --visnet-v2-chromato-feature-dim 3 --visnet-v2-graph-hidden-dim 512 --visnet-v2-physchem-hidden-dim 128 --visnet-v2-toxicity-hidden-dim 64 --visnet-v2-chromato-hidden-dim 32 --visnet-v2-fusion-hidden-dims 512 256 128 --visnet-v2-dropout-rate 0.3 -ds 200 -it 20
```

# Predict

```bash
# neg
python predict.py \
  --model_path ./log/train_20251024-012305_dim48_layerH6_layerO6_batch256_lr0.0001_iter50_debug2000/model.pt \
  --params_path ./log/train_20251024-012305_dim48_layerH6_layerO6_batch256_lr0.0001_iter50_debug2000/training_params.json

# pos
python predict.py \
  --model_path ./log/train_20251024-012645_dim48_layerH6_layerO6_batch256_lr0.0001_iter50_debug2000/model.pt \
  --params_path ./log/train_20251024-012645_dim48_layerH6_layerO6_batch256_lr0.0001_iter50_debug2000/training_params.json
```

## predict-2

```bash
1. pos
python predict_v0.py \
  --model_path ./log/train_20251020-074737_dim48_layerH6_layerO6_batch256_lr0.0001_iter200/model.pt \
  --params_path ./log/train_20251020-074737_dim48_layerH6_layerO6_batch256_lr0.0001_iter200/training_params.json

2. neg
python predict.py \
  --model_path ./log/train_20251020-133515_dim48_layerH6_layerO6_batch256_lr0.0001_iter200/model.pt \
  --params_path ./log/train_20251020-133515_dim48_layerH6_layerO6_batch256_lr0.0001_iter200/training_params.json
```

### predict - visnet
```bash
python predict.py \
  --model_path ./log/train_20251022-180809_dim48_layerH6_layerO6_batch64_lr0.0001_iter200/model.pt \
  --params_path ./log/train_20251022-180809_dim48_layerH6_layerO6_batch64_lr0.0001_iter200/training_params.json
```


### predict - 10.24
```bash
python predict.py \
  --model_path ./log/train_20251024-013405_dim48_layerH6_layerO6_batch256_lr0.0001_iter50_debug3000/model.pt \
  --params_path ./log/train_20251024-013405_dim48_layerH6_layerO6_batch256_lr0.0001_iter50_debug3000/training_params.json

python predict.py \
  --model_path ./log/train_20251024-013904_dim48_layerH6_layerO6_batch256_lr0.0001_iter50_debug3000/model.pt \
  --params_path ./log/train_20251024-013904_dim48_layerH6_layerO6_batch256_lr0.0001_iter50_debug3000/training_params.json

# l1
python predict.py \
  --model_path ./log/train_20251024-014338_dim48_layerH6_layerO6_batch256_lr0.0001_iter50_debug2000/model.pt \
  --params_path ./log/train_20251024-014338_dim48_layerH6_layerO6_batch256_lr0.0001_iter50_debug2000/training_params.json

# INFO Real Predict
python predict.py \
  --model_path ./log/train_20251025-010256_dim48_layerH6_layerO6_batch256_lr0.0001_iter200/model.pt \
  --params_path ./log/train_20251025-010256_dim48_layerH6_layerO6_batch256_lr0.0001_iter200/training_params.json

python predict.py \
  --model_path ./log/train_20251025-010340_dim48_layerH6_layerO6_batch256_lr0.0001_iter200/model.pt \
  --params_path ./log/train_20251025-010340_dim48_layerH6_layerO6_batch256_lr0.0001_iter200/training_params.json


# INFO Predict Original Data
python predict.py \
  --model_path ./log/neg-train_20251025-010256_dim48_layerH6_layerO6_batch256_lr0.0001_iter200/model.pt \
  --params_path ./log/neg-train_20251025-010256_dim48_layerH6_layerO6_batch256_lr0.0001_iter200/training_params.json \
  --files_to_process "MMF-GNN_RTI_neg_Covered_by_Model.csv" \
  --filter_column "no"

python predict.py \
  --model_path ./log/pos-train_20251025-010340_dim48_layerH6_layerO6_batch256_lr0.0001_iter200/model.pt \
  --params_path ./log/pos-train_20251025-010340_dim48_layerH6_layerO6_batch256_lr0.0001_iter200/training_params.json \
  --files_to_process "MMF-GNN_RTI_neg_Covered_by_Model.csv" \
  --filter_column "no"
```

# Train - visnet - v1
```bash
CUDA_VISIBLE_DEVICES=1 python train.py --standardize --model visnet_v1 --loss-function l1 --batch-train 128 -it 50 -ds 4000

CUDA_VISIBLE_DEVICES=0 python train.py --standardize --model visnet_v1 --loss-function l1 --batch-train 32 -it 200
```

## predic

```bash
python predict.py \
  --model_path ./log/train_20251025-130527_dim48_layerH6_layerO6_batch32_lr0.0001_iter200/model.pt \
  --params_path ./log/train_20251025-130527_dim48_layerH6_layerO6_batch32_lr0.0001_iter200/training_params.json \
  --files_to_process "MMF-GNN_RTI_neg_Covered_by_Model.csv" \
  --filter_column "no"
```


# Train - visnet - v1 - dropput
```bash
CUDA_VISIBLE_DEVICES=1 python train.py --standardize --model visnet_v1 --loss-function l1 --batch-train 128 -it 100  --visnet-v1-dropout-rate 0.2

CUDA_VISIBLE_DEVICES=1 python train.py --standardize --model visnet_v1 --loss-function l1 --batch-train 128 -it 100  --visnet-v1-dropout-rate 0.3
```

## basic
```bash
CUDA_VISIBLE_DEVICES=1 python train.py --standardize --loss-function l1 --dropout 0.2 -it 5 -ds 200
```

## extend
```bash
CUDA_VISIBLE_DEVICES=1 python train.py --model extended --feature-config basic,toxicity --standardize --loss-function l1 -it 150
```

# Predict - only

```bash
python predict.py \
  --model_path ./log/neg-train_20251025-010256_dim48_layerH6_layerO6_batch256_lr0.0001_iter200/model.pt \
  --params_path ./log/neg-train_20251025-010256_dim48_layerH6_layerO6_batch256_lr0.0001_iter200/training_params.json \
  --files_to_process "MMF-GNN-valid-smiles.csv" \
  --filter_column "no" \
  --simple_prediction


python predict.py \
  --model_path ./log/pos-train_20251025-010340_dim48_layerH6_layerO6_batch256_lr0.0001_iter200/model.pt \
  --params_path ./log/pos-train_20251025-010340_dim48_layerH6_layerO6_batch256_lr0.0001_iter200/training_params.json \
  --files_to_process "MMF-GNN-valid-smiles.csv" \
  --filter_column "no" \
  --simple_prediction
```