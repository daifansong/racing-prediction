import importlib
import os
import time

import pandas as pd

from ml_util_train import (
    TRAIN_TASKS,
    format_duration,
    load_data,
    odds_auc,
    read_x_features,
    roc_auc,
    test_model,
)

OUT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
MODEL_DIR = os.path.join(OUT_MODEL_PATH, 'dl')
MODEL_NAMES = (
    'mlp_resnet_tower_1',
    'mlp_resnet_tower_2',
    'mlp_resnet_tower_3',
    'mlp_resnet_tower_4',
    'mlp_resnet_tower_5',
)
SCORE_COLUMNS = (
    'auc_win_with_odd',
    'auc_place_with_odd',
    'auc_win_no_odd',
    'auc_place_no_odd',
)


def model_class(model_name):
    return importlib.import_module(f'models.dl.{model_name}').Model


def task_columns(columns, odd_index, x_key):
    if x_key == 'x_no_odd':
        return columns[:odd_index] + columns[odd_index + 1:]
    return list(columns)


def train_dl():
    columns = read_x_features()
    odd_index = columns.index('win_odd')
    train_table, val_table = load_data()
    print(f'odds y_win auc {odds_auc(val_table, "y_win"):.4f}')
    print(f'odds y_place auc {odds_auc(val_table, "y_place"):.4f}')
    os.makedirs(MODEL_DIR, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    score_path = os.path.join(MODEL_DIR, f'p1_dl_train_{stamp}.csv')
    pd.DataFrame(columns=['model', *SCORE_COLUMNS]).to_csv(score_path, index=False)
    print(f'Saving scores: {score_path}')

    for model_name in MODEL_NAMES:
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
            print(f'\n\n\nTraining model: {model_name}, {task_name}, {y_key}, {x_train.shape[1]} columns, {len(y_train)} starters')
            started = time.perf_counter()
            model = model_cls(task_columns(columns, odd_index, x_key))
            model.train(x_train, y_train, x_val, y_val)
            print(f'{model_name} {task_name} train finish, using {format_duration(time.perf_counter() - started)}')
            model.save(os.path.join(model_dir, task_name))
            print(f'Saving model: {os.path.join(model_dir, task_name)}.pt\n')
            score = test_model(model, x_val, y_val)
            task_auc = roc_auc(y_val, score)
            row[f'auc_{task_name}'] = f'{task_auc:.4f}'
            print(f'{model_name} {task_name} val auc {task_auc:.4f}')
        pd.DataFrame([row], columns=['model', *SCORE_COLUMNS]).to_csv(score_path, mode='a', header=False, index=False)


if __name__ == '__main__':
    train_dl()
