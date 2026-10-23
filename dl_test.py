import importlib
import os
import time

import pandas as pd

from ml_test_all import load_split, predict_score, read_x_features
from ml_train_all import odds_auc, roc_auc

OUT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
MODEL_DIR = os.path.join(OUT_MODEL_PATH, 'dl')
MODEL_NAMES = ('mlp',)
SCORE_COLUMNS = (
    'auc_win_with_odd',
    'auc_place_with_odd',
    'auc_win_no_odd',
    'auc_place_no_odd',
)
TASKS = (
    ('win_with_odd', 'x_with_odd', 'score_win_with_odd'),
    ('place_with_odd', 'x_with_odd', 'score_place_with_odd'),
    ('win_no_odd', 'x_no_odd', 'score_win_no_odd'),
    ('place_no_odd', 'x_no_odd', 'score_place_no_odd'),
)


def load_models(model_name):
    model_cls = importlib.import_module(f'models.dl.{model_name}').Model
    model_dir = os.path.join(MODEL_DIR, model_name)
    models = {}
    for task_name, feature_key, score_name in TASKS:
        model_path = os.path.join(model_dir, task_name)
        if not os.path.isfile(model_path + '.pt'):
            raise ValueError(f'\t{task_name} saved failed')
        model = model_cls()
        model.load(model_path)
        models[task_name] = model
    return models


def test_dl():
    columns = read_x_features()
    odd_index = columns.index('win_odd')
    table = load_split('test', columns, odd_index)
    print(f'test {len(table["y_win"])} starters')
    print(f'odds y_win auc {odds_auc(table, "y_win"):.4f}')
    print(f'odds y_place auc {odds_auc(table, "y_place"):.4f}')
    os.makedirs(OUT_MODEL_PATH, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    score_path = os.path.join(OUT_MODEL_PATH, f'p1_dl_test_{stamp}.csv')
    pd.DataFrame(columns=['model', *SCORE_COLUMNS]).to_csv(score_path, index=False)
    print(f'Saving scores: {score_path}')
    for model_name in MODEL_NAMES:
        models = load_models(model_name)
        row = {'model': model_name}
        for task_name, feature_key, score_name in TASKS:
            y_key = 'y_win' if task_name.startswith('win_') else 'y_place'
            score = predict_score(models[task_name], table[feature_key], len(table[y_key]), task_name)
            task_auc = roc_auc(table[y_key], score)
            row[f'auc_{task_name}'] = f'{task_auc:.4f}'
            print(f'{model_name} {task_name} auc {task_auc:.4f}')
        pd.DataFrame([row], columns=['model', *SCORE_COLUMNS]).to_csv(
            score_path, mode='a', header=False, index=False,
        )


if __name__ == '__main__':
    test_dl()
