# 关键CLI命令记录

## 1. 激活conda环境
```bash
conda activate mol-edit
```

## 2. 提取SHAP模型
```bash
cd /home/data2/rhj/project/gnn/gnn-1 && python extract_shap_model.py \
    --model-path /home/data2/rhj/project/gnn/gnn-1/log/neg-train_20251028-110803_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/model.pt \
    --training-params-path /home/data2/rhj/project/gnn/gnn-1/log/neg-train_20251028-110803_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/training_params.json \
    --output-path /home/data2/rhj/project/gnn/gnn-1/log/neg-train_20251028-110803_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/model_shap.pth \
    --physchem-dim 4
```

## 3. 运行SHAP分析
```bash
cd /home/data2/rhj/project/gnn/gnn-1 && python fixed_shap_analysis.py \
    --model-path /home/data2/rhj/project/gnn/gnn-1/log/neg-train_20251028-110803_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/model_shap.pth \
    --training-params-path /home/data2/rhj/project/gnn/gnn-1/log/neg-train_20251028-110803_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/training_params.json \
    --dataset-name MMF_GNN_neg \
    --output-dir /home/data2/rhj/project/gnn/gnn-1/log/neg-train_20251028-110803_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/shap_analysis \
    --sample-size 10
```

## 4. 查看SHAP分析结果
```bash
# 查看生成的文件
ls -la /home/data2/rhj/project/gnn/gnn-1/log/neg-train_20251028-110803_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/shap_analysis/

# 查看特征重要性结果
cat /home/data2/rhj/project/gnn/gnn-1/log/neg-train_20251028-110803_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/shap_analysis/feature_importance.json
```

## SHAP分析结果

根据分析结果，4个物化特征对模型预测的贡献度排序如下：

1. **nRotB**（可旋转键数量）: 15.468710
2. **LogP**（脂水分配系数）: 0.060096
3. **TPSA**（拓扑极性表面积）: 0.001450
4. **MW**（分子量）: 0.001296

结果表明，可旋转键数量(nRotB)是最重要的特征，其重要性远超其他特征。