import importlib
import os
import subprocess
import sys
import time

import pandas as pd

from ml_util_train import (
    OUT_MODEL_PATH,
    SCORE_COLUMNS,
    TRAIN_TASKS,
    format_duration,
    load_data,
    odds_auc,
    print_val_odds_auc,
    roc_auc,
    test_model,
)

MODEL_DIR = os.path.join(OUT_MODEL_PATH, 'ml', 'all')

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

MODEL_NAMES = T1_MODEL_NAMES + T2_MODEL_NAMES + T3_MODEL_NAMES

def model_class(model_name):
    return importlib.import_module(f'models.ml.all.{model_name}').Model

def train_model(model_cls, x_train, y_train, x_val, y_val):
    model = model_cls()
    model.fit(x_train, y_train, x_val, y_val)
    return model

def model_is_saved(model_name):
    model_dir = os.path.join(MODEL_DIR, model_name)
    return all(
        os.path.isfile(os.path.join(model_dir, task_name + '.txt'))
        for task_name in TRAIN_TASKS
    )

def train_one(model_name, score_path):
    train_table, val_table = load_data()
    model_cls = model_class(model_name)
    model_dir = os.path.join(MODEL_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    row = {'model': model_name}
    for task_name, xy_data in TRAIN_TASKS.items():
        x_key = xy_data['x']
        y_key = xy_data['y']
        x_train = train_table[x_key]
        y_train = train_table[y_key]
        x_val = val_table[x_key]
        y_val = val_table[y_key]
        print(f'\n\n\nTraining model: {model_name}, {task_name}, {y_key}, {x_train.shape[1]} columns, {len(y_train)} starters', flush=True)
        started = time.perf_counter()
        model = train_model(model_cls, x_train, y_train, x_val, y_val)
        elapsed = time.perf_counter() - started
        print(f'{model_name} {task_name} train finish, using {format_duration(elapsed)}', flush=True)
        model.save(os.path.join(model_dir, task_name))
        print(f'Saving model: {os.path.join(model_dir, task_name)}.txt\n', flush=True)
        score = test_model(model, x_val, y_val)
        task_auc = roc_auc(y_val, score)
        row[f'auc_{task_name}'] = f'{task_auc:.4f}'
        print(f'{model_name} {task_name} val auc {task_auc:.4f}', flush=True)
    pd.DataFrame([row], columns=['model', *SCORE_COLUMNS]).to_csv(score_path, mode='a', header=False, index=False)

def train_ml():
    if len(sys.argv) >= 4 and sys.argv[1] == '--model':
        train_one(sys.argv[2], sys.argv[3])
        return
    os.makedirs(OUT_MODEL_PATH, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    score_path = os.path.join(OUT_MODEL_PATH, f'p1_ml_train_{stamp}.csv')
    pd.DataFrame(columns=['model', *SCORE_COLUMNS]).to_csv(score_path, index=False)
    print(f'Saving scores: {score_path}', flush=True)
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
                [sys.executable, os.path.abspath(__file__), '--model', model_name, score_path],
                check=True,
            )

if __name__ == '__main__':
    train_ml()
