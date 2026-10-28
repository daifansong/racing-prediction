import os
import subprocess
import sys
import tempfile
import time

import pandas as pd

from ml_train import (
    ALL_MODEL_NAMES,
    T1_MODEL_NAMES,
    T2_MODEL_NAMES,
    T3_MODEL_NAMES,
    load_config,
    params_text,
)
from utils_test import (
    ROW_COLUMNS,
    SCORE_COLUMNS,
    SCORE_DIR,
    append_row,
    load_split,
    model_is_saved,
    odds_metric_row,
    print_label_rates,
    read_x_features,
    test_one,
)

ALL_MODEL_PATH = os.path.join(SCORE_DIR, 'all')
SELECTED_MODEL_PATH = os.path.join(SCORE_DIR, 'selected')
ALL_PACKAGE = 'models.all'
SELECTED_PACKAGE = 'models.selected'
PROBABILITY_MODELS = frozenset({
    'logistic_l2', 'logistic_l1', 'logistic_enet', 'logistic_cv', 'logistic_none',
    'sgd_log_loss', 'sgd_modified_huber', 'xgboost_linear', 'lda', 'qda',
    'gaussian_nb', 'bernoulli_nb', 'multinomial_nb', 'complement_nb', 'categorical_nb',
    'probit', 'firth_logistic', 'bayesian_logistic', 'logistic_gam',
    'rbf_sampler_linear', 'nystroem_linear',
    'decision_tree', 'random_forest', 'extra_trees', 'bagging', 'adaboost', 'gbdt', 'hist_gbdt',
    'lightgbm', 'lightgbm_goss', 'lightgbm_dart', 'lightgbm_rf',
    'xgboost', 'xgboost_dart', 'catboost', 'catboost_crossentropy',
    'ngboost', 'ebm', 'knn', 'radius_neighbors',
    'label_propagation', 'label_spreading',
    'calibrated', 'voting', 'stacking',
    'balanced_random_forest', 'easy_ensemble', 'rusboost', 'balanced_bagging',
})
SELECTED_PROBABILITY_MODELS = frozenset(ALL_MODEL_NAMES)
HISTORY_COLUMNS = (
    'time', 'task', 'params', *SCORE_COLUMNS,
)

def task_params(task, config):
    name = str(task)
    if name.endswith('_with_odd'):
        chosen = config['with_odd']
    elif name.endswith('_no_odd'):
        chosen = config['no_odd']
    else:
        raise ValueError(f'\t{name} saved failed')
    return params_text(chosen)

def history_frame(tested_at, config, scored):
    frame = scored.drop(columns=['model']).copy()
    frame.insert(0, 'time', tested_at)
    frame.insert(2, 'params', frame['task'].map(lambda task: task_params(task, config)))
    return frame[list(HISTORY_COLUMNS)]

def append_history(model_name, frame):
    history_dir = os.path.join(SCORE_DIR, 'selected')
    os.makedirs(history_dir, exist_ok=True)
    path = os.path.join(history_dir, f'{model_name}.csv')
    header = not os.path.isfile(path) or os.path.getsize(path) == 0
    frame.to_csv(path, mode='a', header=header, index=False)
    return path

def run_all():
    print_label_rates()
    columns = read_x_features()
    odd_index = columns.index('win_odd')
    val_table = load_split('val', columns, odd_index)
    test_table = load_split('test', columns, odd_index)
    os.makedirs(ALL_MODEL_PATH, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    score_path = os.path.join(ALL_MODEL_PATH, f'p1_ml_test_{stamp}.csv')
    pd.DataFrame(columns=list(ROW_COLUMNS)).to_csv(score_path, index=False)
    print(f'\nSaving scores: {score_path}', flush=True)
    for task, y_key in (('win', 'y_win'), ('place', 'y_place')):
        row = odds_metric_row(task, y_key, val_table, test_table)
        append_row(score_path, row)
        print(f'odds {task} ' + ' '.join(f'{key} {row[key]}' for key in SCORE_COLUMNS), flush=True)
    for tier_name, tier_models in (
        ('T1', T1_MODEL_NAMES),
        ('T2', T2_MODEL_NAMES),
        ('T3', T3_MODEL_NAMES),
    ):
        print(f'\n\n\nStart {tier_name}', flush=True)
        for model_name in tier_models:
            if not model_is_saved(model_name, ALL_MODEL_PATH):
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
    if not model_is_saved(model_name, SELECTED_MODEL_PATH):
        raise ValueError(f'\t{model_name} saved failed')
    print_label_rates()
    columns = read_x_features()
    odd_index = columns.index('win_odd')
    val_table = load_split('val', columns, odd_index)
    test_table = load_split('test', columns, odd_index)
    for task, y_key in (('win', 'y_win'), ('place', 'y_place')):
        row = odds_metric_row(task, y_key, val_table, test_table)
        print(f'odds {task} ' + ' '.join(f'{key} {row[key]}' for key in SCORE_COLUMNS), flush=True)
    tested_at = time.strftime('%Y%m%d_%H%M%S')
    temp_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
    temp_path = temp_file.name
    temp_file.close()
    try:
        test_one(model_name, temp_path, SELECTED_MODEL_PATH, SELECTED_PACKAGE, SELECTED_PROBABILITY_MODELS)
        scored = pd.read_csv(temp_path, header=None, names=list(ROW_COLUMNS))
    finally:
        os.remove(temp_path)
    history_path = append_history(model_name, history_frame(tested_at, config, scored))
    print(f'\nSaving scores: {history_path}', flush=True)

def main():
    if len(sys.argv) < 2:
        raise ValueError('\tconfig saved failed')
    suite = sys.argv[1]
    if suite == 'all':
        if len(sys.argv) >= 5 and sys.argv[2] == '--model':
            test_one(sys.argv[3], sys.argv[4], ALL_MODEL_PATH, ALL_PACKAGE, PROBABILITY_MODELS)
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
