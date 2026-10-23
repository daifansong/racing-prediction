# AGENTS.md

## 任务目标

寻找 LightGBM 模型的最佳超参数配置，以实现在测试集上验证效果的最优表现。

## 工作约束与规则

1. **只允许修改配置文件**：只能通过修改 `configs/lightgbm.json` 来调整模型参数。
2. **禁止修改代码文件**：严禁修改任何 `.py` 或其他代码文件。
3. **严格遵守执行命令**：
   - **模型训练**：仅使用 `python ml_train_selected.py --model lightgbm`
   - **模型测试**：仅使用 `python ml_test_selected.py --model lightgbm`

## 配置文件格式 (`configs/lightgbm.json`)

配置文件结构如下：

```json
{
  "model": "lightgbm",
  "no_odd": {},
  "with_odd": {}
}
```

请在 `no_odd` 与 `with_odd` 字典中填入要探索和优化的具体超参数（如 `learning_rate`、`num_leaves`、`max_depth`、`n_estimators`、`subsample`、`colsample_bytree` 等）。

## 工作流程

1. **配置参数**

   编写或更新 `configs/lightgbm.json` 中的参数配置。

2. **模型训练**

   运行训练脚本：

   ```bash
   python ml_train_selected.py --model lightgbm
   ```

3. **模型测试与效果查看**

   运行测试脚本：

   ```bash
   python ml_test_selected.py --model lightgbm
   ```

   测试运行完毕后，测试结果会保存在 `out/selected/lightgbm.csv`（或对应输出目录下的 CSV 文件）中，可直接读取该文件评估模型表现。

4. **对比与迭代**

   查看 CSV 文件中的评估指标。根据测试指标反馈调整 `configs/lightgbm.json` 中的参数，重复上述步骤，直到找到最佳配置。
