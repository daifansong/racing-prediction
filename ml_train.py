import importlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from utils_train import (
    OUT_MODEL_PATH,
    SCORE_COLUMNS,
    TRAIN_TASKS,
    format_duration,
    load_data,
    print_val_odds_auc,
    roc_auc,
    test_model,
)

ALL_MODEL_DIR = os.path.join(OUT_MODEL_PATH, 'all')
SELECTED_MODEL_DIR = os.path.join(OUT_MODEL_PATH, 'selected')
ALL_PACKAGE = 'models.all'
SELECTED_PACKAGE = 'models.selected'
ALL_ROW_COLUMNS = ('model', *SCORE_COLUMNS)

T1_MODEL_NAMES = (
    'lightgbm', 'lightgbm_goss', 'lightgbm_dart', 'lightgbm_rf',
    'xgboost', 'xgboost_dart', 'xgboost_hinge',
    'catboost', 'catboost_crossentropy',
    'hist_gbdt', 'random_forest', 'extra_trees', 'logistic_l2',
    'balanced_random_forest', 'easy_ensemble', 'rusboost',
)
T2_MODEL_NAMES = (
    'adaboost', 'gbdt', 'ngboost', 'ebm', 'balanced_bagging',
    'logistic_enet', 'logistic_cv', 'sgd_log_loss',
    'calibrated', 'voting', 'stacking',
)
T3_MODEL_NAMES = (
    'logistic_l1', 'logistic_none', 'ridge', 'ridge_cv',
    'sgd_modified_huber', 'sgd_hinge', 'sgd_squared_hinge',
    'perceptron', 'averaged_perceptron', 'passive_aggressive', 'linear_svc',
    'xgboost_linear', 'lda', 'qda',
    'gaussian_nb', 'bernoulli_nb', 'multinomial_nb', 'complement_nb', 'categorical_nb',
    'probit', 'firth_logistic', 'bayesian_logistic', 'logistic_gam',
    'svc_rbf', 'svc_poly', 'svc_sigmoid',
    'nusvc_rbf', 'nusvc_poly', 'nusvc_sigmoid',
    'rbf_sampler_linear', 'nystroem_linear',
    'decision_tree', 'bagging',
    'knn', 'radius_neighbors', 'nearest_centroid',
    'label_propagation', 'label_spreading'
)

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
    return importlib.import_module(f'{SELECTED_PACKAGE}.{model_name}')

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

def model_is_saved(model_name):
    model_dir = os.path.join(ALL_MODEL_DIR, model_name)
    return all(
        os.path.isfile(os.path.join(model_dir, task_name + '.txt'))
        for task_name in TRAIN_TASKS
    )

def train_one(package, model_root, model_name, score_path, config, columns):
    train_table, val_table = load_data()
    model_cls = importlib.import_module(f'{package}.{model_name}').Model
    model_dir = os.path.join(model_root, model_name)
    os.makedirs(model_dir, exist_ok=True)
    row = {'model': model_name}
    for task_name, xy_data in TRAIN_TASKS.items():
        x_key = xy_data['x']
        y_key = xy_data['y']
        params = None if config is None else config[xy_data['group']]
        x_train = train_table[x_key]
        y_train = train_table[y_key]
        x_val = val_table[x_key]
        y_val = val_table[y_key]
        if config is None:
            print(f'\n\n\nTraining model: {model_name}, {task_name}, {y_key}, {x_train.shape[1]} columns, {len(y_train)} starters', flush=True)
        else:
            print(
                f'\n\n\nTraining model: {model_name}, {task_name}, {y_key}, '
                f'{x_train.shape[1]} columns, {len(y_train)} starters, params {params}',
                flush=True,
            )
        started = time.perf_counter()
        model = model_cls() if config is None else model_cls(params)
        model.fit(x_train, y_train, x_val, y_val)
        elapsed = time.perf_counter() - started
        print(f'{model_name} {task_name} train finish, using {format_duration(elapsed)}', flush=True)
        model.save(os.path.join(model_dir, task_name))
        print(f'Saving model: {os.path.join(model_dir, task_name)}.txt\n', flush=True)
        score = test_model(model, x_val, y_val)
        if config is not None and not np.isfinite(score).all():
            raise ValueError('\tscore saved failed')
        task_auc = roc_auc(y_val, score)
        row[f'auc_{task_name}'] = f'{task_auc:.4f}'
        print(f'{model_name} {task_name} val auc {task_auc:.4f}', flush=True)
    if config is not None:
        row['with_odd_params'] = params_text(config['with_odd'])
        row['no_odd_params'] = params_text(config['no_odd'])
    pd.DataFrame([row], columns=list(columns)).to_csv(score_path, mode='a', header=False, index=False)

def run_all():
    os.makedirs(ALL_MODEL_DIR, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    score_path = os.path.join(ALL_MODEL_DIR, f'p1_ml_train_{stamp}.csv')
    pd.DataFrame(columns=list(ALL_ROW_COLUMNS)).to_csv(score_path, index=False)
    print(f'\nSaving scores: {score_path}', flush=True)
    print_val_odds_auc()
    for tier_name, tier_models in (
        ('T1', T1_MODEL_NAMES),
        ('T2', T2_MODEL_NAMES),
        ('T3', T3_MODEL_NAMES),
    ):
        print(f'\n\n\nStart {tier_name}', flush=True)
        for model_name in tier_models:
            if model_is_saved(model_name):
                print(f'skip {model_name}', flush=True)
                continue
            print(f'\n\n\nStart process: {model_name}', flush=True)
            subprocess.run(
                [sys.executable, os.path.abspath(__file__), 'all', '--model', model_name, score_path],
                check=True,
            )

def run_selected():
    if len(sys.argv) != 4 or sys.argv[2] != '--model' or sys.argv[3] not in ALL_MODEL_NAMES:
        raise ValueError('\tconfig saved failed')
    model_name = sys.argv[3]
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', model_name + '.json')
    config = load_config(config_path)
    if config['model'] != model_name:
        raise ValueError(f'\t{model_name} saved failed')
    log_dir = os.path.join(OUT_MODEL_PATH, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    score_path = os.path.join(log_dir, f'{stamp}_{model_name}.csv')
    pd.DataFrame(columns=list(ROW_COLUMNS)).to_csv(score_path, index=False)
    print(f'\nSaving scores: {score_path}', flush=True)
    print_val_odds_auc()
    train_one(SELECTED_PACKAGE, SELECTED_MODEL_DIR, model_name, score_path, config, ROW_COLUMNS)

def main():
    if len(sys.argv) < 2:
        raise ValueError('\tconfig saved failed')
    suite = sys.argv[1]
    if suite == 'all':
        if len(sys.argv) >= 5 and sys.argv[2] == '--model':
            train_one(ALL_PACKAGE, ALL_MODEL_DIR, sys.argv[3], sys.argv[4], None, ALL_ROW_COLUMNS)
            return
        if len(sys.argv) != 2:
            raise ValueError('\tconfig saved failed')
        run_all()
        return
    if suite == 'selected':
        run_selected()
        return
    raise ValueError('\tconfig saved failed')

if __name__ == '__main__':
    main()
