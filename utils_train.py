import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(ROOT, 'dataset')
SPEC_CSV = os.path.join(ROOT, 'v1.csv')
OUT_MODEL_PATH = os.path.join(ROOT, 'out', 'ml')

MISSING = {'', '--', '---'}
Y_COLUMNS = ['y_win', 'y_place']
TRAIN_TASKS = {
    'win_with_odd': {'x': 'x_with_odd', 'y': 'y_win', 'group': 'with_odd'},
    'place_with_odd': {'x': 'x_with_odd', 'y': 'y_place', 'group': 'with_odd'},
    'win_no_odd': {'x': 'x_no_odd', 'y': 'y_win', 'group': 'no_odd'},
    'place_no_odd': {'x': 'x_no_odd', 'y': 'y_place', 'group': 'no_odd'},
}
SCORE_COLUMNS = (
    'auc_win_with_odd',
    'auc_place_with_odd',
    'auc_win_no_odd',
    'auc_place_no_odd',
)

def read_x_features():
    if not os.path.isfile(SPEC_CSV):
        raise ValueError('\tfeature-v1.csv saved failed')
    feature_df = pd.read_csv(SPEC_CSV, dtype=str, keep_default_na=False)
    return feature_df['feature'].tolist()

def to_np_matrix(frame, columns):
    numeric = frame[columns].replace(list(MISSING), np.nan)
    numeric = numeric.apply(pd.to_numeric, errors='coerce')
    return numeric.to_numpy(dtype=float)

def load_split(split, columns, odd_index):
    x_path = os.path.join(DATASET_DIR, f'x_{split}.csv')
    y_path = os.path.join(DATASET_DIR, f'y_{split}.csv')
    for path in (x_path, y_path):
        if not os.path.isfile(path):
            raise ValueError(f'\t{split} saved failed')
    x_df = pd.read_csv(x_path, dtype=str, keep_default_na=False)
    y_df = pd.read_csv(y_path, dtype=str, keep_default_na=False)
    if list(x_df.columns) != columns or list(y_df.columns) != Y_COLUMNS or len(x_df) != len(y_df):
        raise ValueError(f'\t{split} saved failed')
    x_with = to_np_matrix(x_df, columns)
    return {
        'x_with_odd': x_with,
        'x_no_odd': np.delete(x_with, odd_index, axis=1),
        'y_win': pd.to_numeric(y_df['y_win'], errors='coerce').to_numpy(dtype=float),
        'y_place': pd.to_numeric(y_df['y_place'], errors='coerce').to_numpy(dtype=float),
        'win_odd': pd.to_numeric(x_df['win_odd'], errors='coerce').to_numpy(dtype=float),
    }

def load_data():
    columns = read_x_features()
    odd_index = columns.index('win_odd')
    train_table = load_split('train', columns, odd_index)
    val_table = load_split('val', columns, odd_index)
    print(f'Loading dataset: train {len(train_table["y_win"])} starters', flush=True)
    print(f'Loading dataset: val {len(val_table["y_win"])} starters', flush=True)
    return train_table, val_table

def test_model(ml_model, x_test, y_test):
    test_score = np.asarray(ml_model.predict(x_test), dtype=float).reshape(-1)
    if len(test_score) != len(y_test):
        raise ValueError('\tscore saved failed')
    return test_score

def format_duration(seconds):
    minutes = int(seconds // 60)
    remain = int(seconds % 60)
    return f'{minutes}min {remain}s'

def roc_auc(y_true, y_score):
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError('\tauc saved failed')
    order = np.argsort(y_score, kind='mergesort')
    sorted_scores = y_score[order]
    ranks = np.empty(len(y_score), dtype=float)
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    sum_pos = ranks[y_true == 1].sum()
    return (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

def odds_auc(table, y_key):
    odds = table['win_odd']
    mask = np.isfinite(odds)
    return roc_auc(table[y_key][mask], -odds[mask])

def print_val_odds_auc():
    x_path = os.path.join(DATASET_DIR, 'x_val.csv')
    y_path = os.path.join(DATASET_DIR, 'y_val.csv')
    odds_frame = pd.read_csv(x_path, dtype=str, keep_default_na=False, usecols=['win_odd'])
    y_df = pd.read_csv(y_path, dtype=str, keep_default_na=False)
    odds = pd.to_numeric(odds_frame['win_odd'].replace(list(MISSING), np.nan), errors='coerce').to_numpy(dtype=float)
    y_win = pd.to_numeric(y_df['y_win'], errors='coerce').to_numpy(dtype=float)
    y_place = pd.to_numeric(y_df['y_place'], errors='coerce').to_numpy(dtype=float)
    mask = np.isfinite(odds)
    print(f'odds y_win auc {roc_auc(y_win[mask], -odds[mask]):.4f}', flush=True)
    print(f'odds y_place auc {roc_auc(y_place[mask], -odds[mask]):.4f}', flush=True)
