import os
import subprocess
import sys
import time

import pandas as pd

from ml_train_all import T1_MODEL_NAMES, T2_MODEL_NAMES, T3_MODEL_NAMES
from ml_util_test import (
    ROW_COLUMNS,
    SCORE_COLUMNS,
    SCORE_DIR,
    append_row,
    load_split,
    model_is_saved,
    odds_metric_row,
    predict_score,
    print_label_rates,
    read_x_features,
    test_one,
)

ML_MODEL_PATH = os.path.join(SCORE_DIR, 'ml', 'all')
PACKAGE_NAME = 'all'
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

def test_ml():
    if len(sys.argv) >= 4 and sys.argv[1] == '--model':
        test_one(sys.argv[2], sys.argv[3], ML_MODEL_PATH, PACKAGE_NAME, PROBABILITY_MODELS)
        return
    print_label_rates()
    columns = read_x_features()
    odd_index = columns.index('win_odd')
    val_table = load_split('val', columns, odd_index)
    test_table = load_split('test', columns, odd_index)
    os.makedirs(SCORE_DIR, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    score_path = os.path.join(SCORE_DIR, f'p1_ml_test_{stamp}.csv')
    pd.DataFrame(columns=list(ROW_COLUMNS)).to_csv(score_path, index=False)
    print(f'Saving scores: {score_path}', flush=True)
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
            if not model_is_saved(model_name, ML_MODEL_PATH):
                print(f'skip {model_name}', flush=True)
                continue
            print(f'\n\n\nStart process: {model_name}', flush=True)
            subprocess.run(
                [sys.executable, os.path.abspath(__file__), '--model', model_name, score_path],
                check=True,
            )

if __name__ == '__main__':
    test_ml()
