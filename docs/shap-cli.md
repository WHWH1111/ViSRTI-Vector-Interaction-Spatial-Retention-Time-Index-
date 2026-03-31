# CLI

```bash
# pos-3
python analyze_shap.py `
  --model-path log/data-2/pos-3-train_20251104-055350_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/model.pt `
  --training-params-path log/data-2/pos-3-train_20251104-055350_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/training_params.json `
  --data-path ./data/MMF-3/ `
  --dataset-name MMF_GNN_pos `
  --sample-size 10 `
  --background-samples 1 `
  --batch-size 512 `
  --output-dir ./results/shap/pos-3 `
  --use-train-data

# neg-4
python analyze_shap.py `
  --model-path log/data-2/neg-4-train_20251102-023807_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/model.pt `
  --training-params-path log/data-2/neg-4-train_20251102-023807_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/training_params.json `
  --data-path ./data/MMF-3/ `
  --dataset-name MMF_GNN_neg `
  --sample-size 200 `
  --background-samples 50 `
  --batch-size 512 `
  --output-dir ./results/shap/neg-4 `
  --use-train-data

# neg-3 dim-up
python analyze_shap.py `
  --model-path log/data-2/train_20251112-153539_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/model.pt `
  --training-params-path log/data-2/train_20251112-153539_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/training_params.json `
  --data-path ./data/MMF-3/ `
  --dataset-name MMF_GNN_neg `
  --sample-size 100 `
  --background-samples 50 `
  --batch-size 256 `
  --output-dir ./results/shap/neg-3-dim-up `
  --use-train-data

```

## shap filter by tag

```bash
python analyze_shap.py \
  --model-path 'log/visnet-v2/pos-3-mask(koc)-train_20251126-172959_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/best.pt' \
  --training-params-path 'log/visnet-v2/pos-3-mask(koc)-train_20251126-172959_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/training_params.json' \
  --data-path ./data/MMF-3/ \
  --dataset-name MMF_GNN_pos \
  --sample-size 150 \
  --background-samples 20 \
  --batch-size 1024 \
  --output-dir ./results/shap/pos-3-test \
  --use-train-data \
  --filter-by-tag Aromatic
```


# History Log

## Part I

```bash
CUDA_VISIBLE_DEVICES=2 python analyze_shap.py --sample-size 500 --background-samples 50 --batch-size 64 --use-train-data

CUDA_VISIBLE_DEVICES=2 python analyze_shap.py --sample-size 500 --background-samples 50 --batch-size 64 --use-train-data

CUDA_VISIBLE_DEVICES=2 python analyze_shap.py --sample-size 1000 --background-samples 100 --batch-size 64 --use-train-data

CUDA_VISIBLE_DEVICES=2 python analyze_shap.py --sample-size 1500 --background-samples 150 --batch-size 128 --use-train-data

CUDA_VISIBLE_DEVICES=2 python analyze_shap.py --sample-size 4000 --background-samples 200 --batch-size 128 --use-train-data
```

## Part II

```bash
python analyze_shap.py   --model-path ./log/neg-4-train_20251102-023807_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/model.pt   --training-params-path ./log/neg-4-train_20251102-023807_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/training_params.json   --data-path ./data/MMF-3/   --dataset-name MMF_GNN_neg   --sample-size 500   --background-samples 100   --output-dir ./shap_analysis_results/neg_4   --use-train-data

# 对于物化+毒性+色谱模型 (neg-4):
python analyze_shap.py \
  --model-path ./log/neg-4-train_20251102-023807_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/model.pt \
  --training-params-path ./log/neg-4-train_20251102-023807_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/training_params.json \
  --data-path ./data/MMF-3/ \
  --dataset-name MMF_GNN_neg \
  --sample-size 500 \
  --background-samples 100 \
  --output-dir ./shap_analysis_results/neg_4 \
  --use-train-data \
  --batch-size 128

python analyze_shap.py `
  --model-path ./log/neg-4-train_20251102-023807_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/model.pt `
  --training-params-path ./log/neg-4-train_20251102-023807_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/training_params.json `
  --data-path ./data/MMF-3/ `
  --dataset-name MMF_GNN_neg `
  --sample-size 500 `
  --background-samples 50 `
  --output-dir ./shap_analysis_results/neg_4 `
  --use-train-data `
  --batch-size 512

# 对于物化+毒性模型 (pos-3):
python analyze_shap.py \
  --model-path ./log/pos-3-train_20251104-055350_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/model.pt \
  --training-params-path ./log/pos-3-train_20251104-055350_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/training_params.json \
  --data-path ./data/MMF-3/ \
  --dataset-name MMF_GNN_pos \
  --sample-size 50 \
  --background-samples 30 \
  --output-dir ./shap_analysis_results/pos_3

python analyze_shap.py `
  --model-path ./log/pos-3-train_20251104-055350_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/model.pt `
  --training-params-path ./log/pos-3-train_20251104-055350_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/training_params.json `
  --data-path ./data/MMF-3/ `
  --dataset-name MMF_GNN_pos `
  --sample-size 500 `
  --background-samples 100 `
  --output-dir ./shap_analysis_results/pos_3 `
  --use-train-data `
  --batch-size 512
```

```bash
python analyze_shap.py `
  --model-path ./log/train_20251110-210035_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/model.pt `
  --training-params-path ./log/train_20251110-210035_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/training_params.json `
  --data-path ./data/MMF-3/ `
  --dataset-name MMF_GNN_neg `
  --sample-size 500 `
  --background-samples 100 `
  --output-dir ./shap_analysis_results/neg_4 `
  --use-train-data `
  --batch-size 512
```