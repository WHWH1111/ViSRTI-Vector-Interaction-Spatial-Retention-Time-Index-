# Train

```bash
# pos
python train_visnet_v2.py --visnet-v2-feature-level graph_physchem_toxicity --standardize --loss-function l1 --batch-train 64 --standardize-features  --standardize --standardize-features --visnet-v2-physchem-mask 1 1 1 0

# neg
python train_visnet_v2.py  --visnet-v2-feature-level graph_physchem_toxicity --standardize --loss-function l1 --batch-train 64 --batch-test 64 --standardize-features --visnet-v2-physchem-mask 1 1 1 0  --visnet-v2-toxicity-mask 1 0 1 1 --standardize --standardize-features -it 100

python train_visnet_v2.py  --visnet-v2-feature-level graph_physchem_toxicity --standardize --loss-function l1 --batch-train 64 --batch-test 64 --standardize-features --visnet-v2-physchem-mask 1 0 1 1 --standardize --standardize-features -it 100

python train_visnet_v2.py --visnet-v2-feature-level graph --standardize --loss-function l1 --batch-train 64 --batch-test 64 --standardize-features --standardize --standardize-features -it 100
```

## Train with 5 fold - 五折训练

```bash
## pos
### 1. 划分数据
python cross_validation.py --dataname MMF_GNN_neg

### 2. 训练
python train_visnet_v2_5.py --cross-validation --visnet-v2-feature-level graph_physchem_toxicity --standardize --loss-function l1 --batch-train 64 --standardize-features --visnet-v2-physchem-mask 1 1 1 0 --cv-dataname MMF_GNN_neg

## neg

```

# Predict

## on data1028_sheet0.csv

```bash
# pos
python predict_visnet_v2.py `
  --model_path './log/visnet-v2/pos-3-mask(koc)-train_20251126-172959_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/best.pt' `
  --params_path './log/visnet-v2/pos-3-mask(koc)-train_20251126-172959_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/training_params.json' `
  --output_dir ./predictions/ `
  --batch_size 512 `
  --input_file 'data\MMF-4\norman数据库.csv'
  --input_file 'D:\Projects\python\gnn-rt-1\data\MMF-4\task-2-full.csv'
  --input_file 'D:\Projects\python\gnn-rt-1\data\MMF-3\data1028_sheet0.csv'
```

```bash
python predict_visnet_v2.py `
  --model_path './log/visnet-v2/neg-3-mask(koc)-train_20251128-162233_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/best.pt' `
  --params_path './log/visnet-v2/neg-3-mask(koc)-train_20251128-162233_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/training_params.json' `
  --input_file 'D:\Projects\python\gnn-rt-1\data\MMF-3\data1028_sheet0.csv' `
  --output_dir ./predictions/ `
  --batch_size 512  `
  --input_file 'D:\Projects\python\gnn-rt-1\data\MMF-4\task-2-full.csv'
  --input_file 'data\MMF-4\norman数据库.csv'
  --input_file 'D:\Projects\python\gnn-rt-1\data\MMF-3\data1028_sheet0.csv'
```
> Using cache file: D:\Projects\python\gnn-rt-1\utils\..\.cache\molecule_graphs\visnet_predict_v2_norman数据库.pt

## on train/val-dataset

```bash
# pos
python predict_visnet_v2.py `
  --model_path 'log/visnet-v2/pos-3-mask(koc)-train_20251126-172959_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/best.pt' `
  --params_path 'log/visnet-v2/pos-3-mask(koc)-train_20251126-172959_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/training_params.json' `
  --input_file 'data/MMF-3/MMF_GNN_pos.csv' `
  --output_dir ./predictions/visnet-v2/ `
  --batch_size 64 `
  --dataset_type test

# neg
python predict_visnet_v2.py `
  --model_path 'log/visnet-v2/neg-3-mask(koc)-train_20251128-162233_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/model.pt' `
  --params_path 'log/visnet-v2/neg-3-mask(koc)-train_20251128-162233_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/training_params.json' `
  --input_file 'data/MMF-3/MMF_GNN_neg.csv' `
  --output_dir ./predictions/visnet-v2/ `
  --batch_size 64 `
  --dataset_type test
```

### add ori model

```bash
python predict_visnet_v2.py `
  --model_path 'log/visnet-v2/pos-1-train_20251104-055250_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/model.pt' `
  --params_path 'log/visnet-v2/pos-1-train_20251104-055250_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/training_params.json' `
  --input_file 'data/MMF-3/MMF_GNN_pos.csv' `
  --output_dir ./predictions/visnet-v2/ `
  --batch_size 64 `
  --dataset_type test

python predict_visnet_v2.py `
  --model_path 'log/visnet-v2/train_20251129-145238_dim48_layerH6_layerO6_batch64_lr0.0001_iter100/model.pt' `
  --params_path 'log/visnet-v2/train_20251129-145238_dim48_layerH6_layerO6_batch64_lr0.0001_iter100/training_params.json' `
  --input_file 'data/MMF-3/MMF_GNN_neg.csv' `
  --output_dir ./predictions/visnet-v2/ `
  --batch_size 64 `
  --dataset_type test
```

## on double-val

```bash
python predict_visnet_v2.py `
  --model_path 'log/visnet-v2/neg-3-mask(koc)-train_20251128-162233_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/best.pt' `
  --params_path 'log/visnet-v2/neg-3-mask(koc)-train_20251128-162233_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/training_params.json' `
  --input_file "data/MMF-3/MMF_GNN_neg_test.csv" `
  --output_dir ./predictions/visnet-v2/ `
  --batch_size 16
```

## Predict with Conf

> dataset_type: test | train

```bash
# pos
python predict_visnet_v2_with_confidence.py `
  --model_path './log/visnet-v2/pos-3-mask(koc)-train_20251126-172959_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/model.pt' `
  --params_path './log/visnet-v2/pos-3-mask(koc)-train_20251126-172959_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/training_params.json' `
  --input_file 'data/MMF-3/MMF_GNN_pos.csv' `
  --output_dir ./predictions/with-conf `
  --dataset_type test `
  --batch_size 64 `
  --n_iterations 100

# neg
python predict_visnet_v2_with_confidence.py `
  --model_path './log/visnet-v2/neg-3-mask(koc)-train_20251128-162233_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/model.pt' `
  --params_path './log/visnet-v2/neg-3-mask(koc)-train_20251128-162233_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/training_params.json' `
  --input_file 'data/MMF-3/MMF_GNN_neg.csv' `
  --output_dir ./predictions/with-conf `
  --batch_size 256 `
  --n_iterations 100 `
  --dataset_type train `
  --max_items 100
```

## Predict with 5 fold

```bash
## pos
python predict_visnet_v2.py `
  --model_path 'log/visnet-v2-5/pos/train_20251130-234211_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/best.pt' `
  --params_path 'log/visnet-v2-5/pos/train_20251130-234211_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/training_params.json' `
  --input_file 'data/MMF-3/MMF_GNN_pos.csv' `
  --output_dir ./predictions/visnet-v2-5fold/pos `
  --batch_size 64 `
  --dataset_file 'data/MMF-3/MMF_GNN_pos_5/fold_1/fold_1_test_set.txt' `
  --cache_name 'visnet-v2-5fold-pos'

python scripts/batch_predict_visnet_v2.py --model_root log/visnet-v2-5/pos --data_root data/MMF-3/MMF_GNN_pos_5 --ion_type pos

python scripts/data-merge/merge_predictions_with_tags.py

python scripts/organize_predictions_and_residuals.py

## neg
python scripts/batch_predict_visnet_v2.py --model_root log/visnet-v2-5/neg --data_root data/MMF-3/MMF_GNN_neg_5 --ion_type neg

python scripts/data-merge/merge_predictions_with_tags.py --predictions_dir predictions/visnet-v2-5fold/neg --tags_file data/MMF-3/tags/neg_compound_tags.csv --output_file predictions/visnet-v2-5fold/neg/merged_predictions_with_tags.csv

python scripts/organize_predictions_and_residuals.py --input_file predictions/visnet-v2-5fold/neg/merged_predictions_with_tags.csv
```

# gnn-rt

## predict

```bash
# pos
python predict.py `
  --model_path "log/gnn-rt/gnn-rt-pos-train_20251028-061259_dim48_layerH6_layerO6_batch256_lr0.0001_iter150/model.pt" `
  --params_path "log/gnn-rt/gnn-rt-pos-train_20251028-061259_dim48_layerH6_layerO6_batch256_lr0.0001_iter150/training_params.json" `
  --input_dir "data/MMF-3" `
  --dict_dir "data/MMF-3" `
  --dataset_type "test"

# neg
python predict.py `
  --model_path "log/gnn-rt/gnn-rt-neg-train_20251028-085037_dim48_layerH6_layerO6_batch256_lr0.0001_iter150/model.pt" `
  --params_path "log/gnn-rt/gnn-rt-neg-train_20251028-085037_dim48_layerH6_layerO6_batch256_lr0.0001_iter150/training_params.json" `
  --input_dir "data/MMF-3" `
  --dict_dir "data/MMF-3" `
  --dataset_type "test"
```