import importlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from ml_util_train import (
    OUT_MODEL_PATH,
    SCORE_COLUMNS,
    TRAIN_TASKS,
    format_duration,
    load_data,
    print_val_odds_auc,
    roc_auc,
    test_model,
)

MODEL_DIR = os.path.join(OUT_MODEL_PATH, 'ml', 'selected')

ALL_MODEL_NAMES = ('lightgbm', 'xgboost', 'catboost', 'gbdt', 'bagging', 'ngboost', 'ebm', 'hist_gbdt', 'random_forest', 'extra_trees')

# 这两项不在 default_params 里，由 fit 按该任务的训练集标签计算。
EXTRA_PARAM_NAMES = {
    'lightgbm': ('pos_weight_mult',),
    'xgboost': ('pos_weight_mult',),
    'catboost': ('auto_class_weights',),
}

ROW_COLUMNS = ('model', *SCORE_COLUMNS, 'with_odd_params', 'no_odd_params')

def model_module(model_name):
    if model_name not in ALL_MODEL_NAMES:
        raise ValueError(f'\t{model_name} saved failed')
    return importlib.import_module(f'models.selected.{model_name}')

def load_config(config_path):
    if not os.path.isfile(config_path):
        raise ValueError('\tconfig saved failed')
    with open(config_path, encoding='utf-8') as handle:
        config = json.load(handle)
    if not isinstance(config, dict) or set(config) != {'model', 'with_odd', 'no_odd'}:
        raise ValueError('\tconfig saved failed')
    model_name = config['model']
    if model_name not in ALL_MODEL_NAMES:
        raise ValueError(f'\t{model_name} saved failed')
    module = model_module(model_name)
    allowed = set(module.default_params()) | set(EXTRA_PARAM_NAMES.get(model_name, ()))
    for group in ('with_odd', 'no_odd'):
        params = config[group]
        if not isinstance(params, dict):
            raise ValueError('\tconfig saved failed')
        for key, value in params.items():
            if key not in allowed or isinstance(value, (dict, list, bool)):
                raise ValueError(f'\t{key} saved failed')
    return config

def params_text(params):
    return json.dumps(params, ensure_ascii=False, sort_keys=True)

def train_one(config, score_path):
    model_name = config['model']
    train_table, val_table = load_data()
    model_cls = model_module(model_name).Model
    model_dir = os.path.join(MODEL_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    row = {'model': model_name}
    for task_name, xy_data in TRAIN_TASKS.items():
        x_key = xy_data['x']
        y_key = xy_data['y']
        params = config[xy_data['group']]
        x_train = train_table[x_key]
        y_train = train_table[y_key]
        x_val = val_table[x_key]
        y_val = val_table[y_key]
        print(
            f'\n\n\nTraining model: {model_name}, {task_name}, {y_key}, '
            f'{x_train.shape[1]} columns, {len(y_train)} starters, params {params}',
            flush=True,
        )
        started = time.perf_counter()
        model = model_cls(params)
        model.fit(x_train, y_train, x_val, y_val)
        elapsed = time.perf_counter() - started
        print(f'{model_name} {task_name} train finish, using {format_duration(elapsed)}', flush=True)
        model.save(os.path.join(model_dir, task_name))
        print(f'Saving model: {os.path.join(model_dir, task_name)}.txt\n', flush=True)
        score = test_model(model, x_val, y_val)
        if not np.isfinite(score).all():
            raise ValueError('\tscore saved failed')
        task_auc = roc_auc(y_val, score)
        row[f'auc_{task_name}'] = f'{task_auc:.4f}'
        print(f'{model_name} {task_name} val auc {task_auc:.4f}', flush=True)
    row['with_odd_params'] = params_text(config['with_odd'])
    row['no_odd_params'] = params_text(config['no_odd'])
    pd.DataFrame([row], columns=list(ROW_COLUMNS)).to_csv(score_path, mode='a', header=False, index=False)

def train_ml():
    if len(sys.argv) != 3 or sys.argv[1] != '--model' or sys.argv[2] not in ALL_MODEL_NAMES:
        raise ValueError('\tconfig saved failed')
    model_name = sys.argv[2]
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', model_name + '.json')
    config = load_config(config_path)
    if config['model'] != model_name:
        raise ValueError(f'\t{model_name} saved failed')
    log_dir = os.path.join(OUT_MODEL_PATH, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    score_path = os.path.join(log_dir, f'{stamp}_{model_name}.csv')
    pd.DataFrame(columns=list(ROW_COLUMNS)).to_csv(score_path, index=False)
    print(f'Saving scores: {score_path}', flush=True)
    print_val_odds_auc()
    train_one(config, score_path)

if __name__ == '__main__':
    train_ml()
