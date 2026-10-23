# AGENTS.md

## 本轮模型

本轮只优化一个模型。模型名由对话开始时的用户指定，记为 `{model}`。

可选名字只有：

`lightgbm`、`xgboost`、`catboost`、`gbdt`、`bagging`、`ngboost`、`ebm`、`hist_gbdt`、`random_forest`、`extra_trees`

在仓库根目录执行，解释器使用 `venv/bin/python`。一次只跑一个进程，训练退出后再测试，测试退出后再读结果。

- 配置文件：`configs/{model}.json`
- 训练：`venv/bin/python ml_train_selected.py --model {model}`
- 测试：`venv/bin/python ml_test_selected.py --model {model}`
- 测试成绩追加在 `out/selected/{model}.csv`
- 训练日志在 `out/logs/{时间}_{model}.csv`

## 工作约束

1. 只创建或修改 `configs/{model}.json`。
2. 不修改任何 `.py` 或其他代码文件，也不改 `v1.csv`、`dataset/` 和已有模型文件。
3. 命令只使用上面的两条。不要换模型名，不要并行启动第二个训练或测试。
4. 每次训练都会完成四个任务，不能只训其中一个：
   - `win_with_odd`、`place_with_odd` 使用 `with_odd`，132 列，含 `win_odd`
   - `win_no_odd`、`place_no_odd` 使用 `no_odd`，131 列，已去掉 `win_odd`
5. 参数名只从本轮模型的代码里取。不要把上一轮模型的键抄进本轮配置。

## 配置文件

顶层只能有这三个键，且 `model` 必须等于本轮名字：

```json
{
  "model": "{model}",
  "with_odd": {},
  "no_odd": {}
}
```

`with_odd` 和 `no_odd` 里没写的键使用代码默认值。值只能是数字、字符串或 `null`。布尔、列表和对象会被拒绝。代码里默认是 `true` 或 `false` 的键不要写进 JSON。`null` 表示 Python 的 `None`；`random_forest`、`extra_trees` 和 `bagging` 的 `max_depth` 用 `null` 表示不限深度。

`lightgbm`、`xgboost`、`catboost` 已有合法的空配置。`configs/gbdt.json` 现在是空文件，不能解析。`bagging`、`ngboost`、`ebm`、`hist_gbdt`、`random_forest`、`extra_trees` 还没有配置文件。文件缺失或不是合法 JSON 时，先写成上面的空骨架，再开始调参。

配置检查失败时，`{键名} saved failed` 表示这个键不在允许名单里，或值的类型不合法。`config saved failed` 表示文件缺失、JSON 无法解析，或顶层键不对。这些都要用配置修正，不要改代码。

## 如何确定合法参数

开调之前先读两处：

1. `models/ml/selected/{model}.py` 里的 `default_params()`
2. `ml_train_selected.py` 里的 `EXTRA_PARAM_NAMES`

这两处的并集才是本轮允许写入的键。`default_params()` 里的值就是默认值。

以下情况读默认字典看不出来：

- `lightgbm` 和 `xgboost` 可以额外写 `pos_weight_mult`。它会乘上该任务训练集的负正比。训练集独赢正例约 8%，前三约 24%。
- `catboost` 的额外键是 `auto_class_weights`，值为字符串。不写这一键表示关闭。
- `lightgbm` 的 `num_boost_round` 和 `early_stopping_rounds` 可以调。`xgboost` 的最大轮数 1000 和早停 50 写在代码里，配置改不了。
- 含赔率任务上对 `win_odd` 的单调约束由模型代码设置。配置里不能写列表，所以不要添加 `monotone_constraints`。
- 随机种子的键名可能是 `seed`、`random_state` 或 `random_seed`。本轮不要改种子。有的模型把种子直接写在代码里，那种也不要改。
- `objective`、`loss_function`、`eval_metric` 保持默认，维持二分类。

## 评价指标

用测试成绩排名，不用训练日志里的验证集 AUC。训练日志只有验证集 AUC，列是 `model`、`auc_win_with_odd`、`auc_place_with_odd`、`auc_win_no_odd`、`auc_place_no_odd`、`with_odd_params`、`no_odd_params`。

测试文件每次追加 4 行。列是 `time`、`task`、`params`、`auc_roc`、`auc_pr`、`logloss`、`brier`、`f1`、`precision`、`recall`、`threshold`、`precision_at_base`。只读本次运行新追加的 4 行，用 stdout 里的 `Saving scores:` 确认路径。

主指标是四个任务的 `auc_roc`，越高越好。`auc_pr`、`f1`、`precision_at_base` 越高越好，`logloss`、`brier` 越低越好。测试脚本还会打印赔率基线，当前数据上独赢 `auc_roc` 约 0.7729，前三约 0.7591。

## 工作流程

1. 确认 `{model}`，读该模型的 `default_params()` 和 `EXTRA_PARAM_NAMES`。配置文件不可用时先写入空骨架。
2. 更新 `configs/{model}.json`。每轮只改少数几个参数。`out/selected/{model}.csv` 里已经出现过的参数组合不要重跑。
3. 运行训练命令，等待进程退出。
4. 运行测试命令，等待进程退出。读取本次追加的 4 行。
5. 与上一轮最好的 4 行比较。本轮更差就把配置退回上一轮最好的 JSON。
6. 重复第 2 步到第 5 步，直到达到停止条件。

## 停止条件

试验次数上限为 8 到 12 次。连续 3 次没有提高四个任务的 `auc_roc` 时停止。

结束后在回复里留下最佳的 `configs/{model}.json` 全文，以及对应的 4 行测试指标。
