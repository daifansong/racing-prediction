import os
import sys
import tempfile
import time

import pandas as pd

from ml_train_selected import ALL_MODEL_NAMES, load_config, params_text
from ml_util_test import (
    ROW_COLUMNS,
    SCORE_COLUMNS,
    SCORE_DIR,
    load_split,
    model_is_saved,
    odds_metric_row,
    print_label_rates,
    read_x_features,
    test_one,
)

ML_MODEL_PATH = os.path.join(SCORE_DIR, 'ml', 'selected')
PACKAGE_NAME = 'selected'
PROBABILITY_MODELS = frozenset(ALL_MODEL_NAMES)
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

def test_ml():
    if len(sys.argv) != 3 or sys.argv[1] != '--model' or sys.argv[2] not in ALL_MODEL_NAMES:
        raise ValueError('\tconfig saved failed')
    model_name = sys.argv[2]
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', model_name + '.json')
    config = load_config(config_path)
    if config['model'] != model_name:
        raise ValueError(f'\t{model_name} saved failed')
    if not model_is_saved(model_name, ML_MODEL_PATH):
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
        test_one(model_name, temp_path, ML_MODEL_PATH, PACKAGE_NAME, PROBABILITY_MODELS)
        scored = pd.read_csv(temp_path, header=None, names=list(ROW_COLUMNS))
    finally:
        os.remove(temp_path)
    history_path = append_history(model_name, history_frame(tested_at, config, scored))
    print(f'Saving scores: {history_path}', flush=True)

if __name__ == '__main__':
    test_ml()
