import importlib
import os

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(ROOT, 'dataset')
SPEC_CSV = os.path.join(ROOT, 'v1.csv')
SCORE_DIR = os.path.join(ROOT, 'out', 'ml')
MISSING = {'', '--', '---'}
Y_COLUMNS = ['y_win', 'y_place']
TASKS = (
    ('win_with_odd', 'x_with_odd'),
    ('place_with_odd', 'x_with_odd'),
    ('win_no_odd', 'x_no_odd'),
    ('place_no_odd', 'x_no_odd'),
)
SCORE_COLUMNS = (
    'auc_roc', 'auc_pr', 'logloss', 'brier',
    'f1', 'precision', 'recall', 'threshold', 'precision_at_base',
)
ROW_COLUMNS = ('model', 'task', *SCORE_COLUMNS)

def read_x_features():
    if not os.path.isfile(SPEC_CSV):
        raise ValueError('\tfeature-v1.csv saved failed')
    feature_df = pd.read_csv(SPEC_CSV, dtype=str, keep_default_na=False)
    names = feature_df['feature'].tolist()
    if 'win_odd' not in names:
        raise ValueError('\tfeature-v1.csv saved failed')
    return names

def to_matrix(frame, columns):
    numeric = frame[columns].replace(list(MISSING), np.nan)
    numeric = numeric.apply(pd.to_numeric, errors='coerce')
    return numeric.to_numpy(dtype=float)

def load_models(model_name, model_root, package_name):
    model_cls = importlib.import_module(
        f'{package_name}.{model_name}'
    ).Model
    model_dir = os.path.join(model_root, model_name)
    models = {}
    for task_name, _feature_key in TASKS:
        model_path = os.path.join(model_dir, task_name)
        if not os.path.isfile(model_path + '.txt'):
            raise ValueError(f'\t{task_name} saved failed')
        model = model_cls()
        model.load(model_path)
        models[task_name] = model
    return models

def predict_score(model, x, starter_count, task_name):
    y_pred = model.predict(x)
    score = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(score) != starter_count or not np.isfinite(score).all():
        raise ValueError(f'\t{task_name} saved failed')
    return score

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
    x_with = to_matrix(x_df, columns)
    return {
        'x_with_odd': x_with,
        'x_no_odd': np.delete(x_with, odd_index, axis=1),
        'y_win': pd.to_numeric(y_df['y_win'], errors='coerce').to_numpy(dtype=float),
        'y_place': pd.to_numeric(y_df['y_place'], errors='coerce').to_numpy(dtype=float),
        'win_odd': pd.to_numeric(x_df['win_odd'], errors='coerce').to_numpy(dtype=float),
    }

def print_label_rates():
    for split in ('train', 'val', 'test'):
        y_path = os.path.join(DATASET_DIR, f'y_{split}.csv')
        if not os.path.isfile(y_path):
            raise ValueError(f'\t{split} saved failed')
        y_df = pd.read_csv(y_path, dtype=str, keep_default_na=False)
        if list(y_df.columns) != Y_COLUMNS:
            raise ValueError(f'\t{split} saved failed')
        y_win = pd.to_numeric(y_df['y_win'], errors='coerce').to_numpy(dtype=float)
        y_place = pd.to_numeric(y_df['y_place'], errors='coerce').to_numpy(dtype=float)
        print(
            f'{split} {len(y_win)} starters, y_win {np.mean(y_win == 1):.4f}, y_place {np.mean(y_place == 1):.4f}',
            flush=True,
        )

def average_precision(y_true, y_score):
    y_true = np.asarray(y_true, dtype=float)
    order = np.argsort(-np.asarray(y_score, dtype=float), kind='mergesort')
    y_sorted = y_true[order]
    hit = y_sorted == 1
    n_pos = int(hit.sum())
    if n_pos == 0:
        raise ValueError('\tauc saved failed')
    tp = np.cumsum(hit)
    precision = tp / np.arange(1, len(y_sorted) + 1)
    return float(precision[hit].mean())

def precision_recall_f1(y_true, y_pred):
    y_true = np.asarray(y_true) == 1
    y_pred = np.asarray(y_pred) == 1
    tp = int(np.sum(y_pred & y_true))
    fp = int(np.sum(y_pred & ~y_true))
    fn = int(np.sum(~y_pred & y_true))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1

def best_f1_threshold(y_true, prob):
    candidates = np.quantile(prob, np.linspace(0.01, 0.99, 99))
    best_threshold = float(candidates[0])
    best_f1 = -1.0
    for threshold in candidates:
        _, _, f1 = precision_recall_f1(y_true, prob >= threshold)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold

def precision_at_base(y_true, y_score):
    y_true = np.asarray(y_true, dtype=float)
    k = int(np.sum(y_true == 1))
    if k == 0:
        raise ValueError('\tauc saved failed')
    order = np.argsort(-np.asarray(y_score, dtype=float), kind='mergesort')
    return float(np.mean(y_true[order[:k]] == 1))

def log_loss(y_true, prob):
    p = np.clip(np.asarray(prob, dtype=float), 1e-15, 1 - 1e-15)
    y = np.asarray(y_true, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def brier_score(y_true, prob):
    p = np.clip(np.asarray(prob, dtype=float), 1e-15, 1 - 1e-15)
    y = np.asarray(y_true, dtype=float)
    return float(np.mean((p - y) ** 2))

def to_probability(model_name, val_score, y_val, test_score, probability_models):
    if model_name in probability_models:
        return val_score, test_score
    if len(np.unique(val_score)) < 2:
        rate = float(np.mean(y_val == 1))
        return np.full_like(val_score, rate, dtype=float), np.full_like(test_score, rate, dtype=float)
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(val_score, y_val)
    return iso.predict(val_score), iso.predict(test_score)

def metric_values(y_val, val_prob, y_test, test_score, test_prob):
    threshold = best_f1_threshold(y_val, val_prob)
    precision, recall, f1 = precision_recall_f1(y_test, test_prob >= threshold)
    values = {
        'auc_roc': roc_auc_score(y_test, test_score),
        'auc_pr': average_precision(y_test, test_score),
        'logloss': log_loss(y_test, test_prob),
        'brier': brier_score(y_test, test_prob),
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'threshold': threshold,
        'precision_at_base': precision_at_base(y_test, test_score),
    }
    return {key: f'{value:.4f}' for key, value in values.items()}

def odds_metric_row(task, y_key, val_table, test_table):
    val_mask = np.isfinite(val_table['win_odd'])
    test_mask = np.isfinite(test_table['win_odd'])
    row = {'model': 'odds', 'task': task}
    row.update(metric_values(
        val_table[y_key][val_mask],
        1.0 / val_table['win_odd'][val_mask],
        test_table[y_key][test_mask],
        -test_table['win_odd'][test_mask],
        1.0 / test_table['win_odd'][test_mask],
    ))
    return row

def append_row(score_path, row):
    pd.DataFrame([row], columns=list(ROW_COLUMNS)).to_csv(score_path, mode='a', header=False, index=False)

def model_is_saved(model_name, model_root):
    model_dir = os.path.join(model_root, model_name)
    return all(
        os.path.isfile(os.path.join(model_dir, task_name + '.txt'))
        for task_name, _feature_key in TASKS
    )

def test_one(model_name, score_path, model_root, package_name, probability_models):
    columns = read_x_features()
    odd_index = columns.index('win_odd')
    val_table = load_split('val', columns, odd_index)
    test_table = load_split('test', columns, odd_index)
    models = load_models(model_name, model_root, package_name)
    model_dir = os.path.join(model_root, model_name)
    rows = []
    for task_name, feature_key in TASKS:
        y_key = 'y_win' if task_name.startswith('win_') else 'y_place'
        val_score = predict_score(models[task_name], val_table[feature_key], len(val_table[y_key]), task_name)
        test_score = predict_score(models[task_name], test_table[feature_key], len(test_table[y_key]), task_name)
        np.save(os.path.join(model_dir, task_name + '.val.npy'), val_score)
        np.save(os.path.join(model_dir, task_name + '.test.npy'), test_score)
        val_prob, test_prob = to_probability(
            model_name, val_score, val_table[y_key], test_score, probability_models,
        )
        row = {'model': model_name, 'task': task_name}
        row.update(metric_values(
            val_table[y_key], val_prob,
            test_table[y_key], test_score, test_prob,
        ))
        print(f'{model_name} {task_name} ' + ' '.join(f'{key} {row[key]}' for key in SCORE_COLUMNS), flush=True)
        rows.append(row)
    pd.DataFrame(rows, columns=list(ROW_COLUMNS)).to_csv(score_path, mode='a', header=False, index=False)
