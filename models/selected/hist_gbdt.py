import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

# sklearn 直方图 GBDT。不用内部验证集做早停。缺失值直接传入。
def default_params():
    return {
        'learning_rate': 0.05,
        'max_iter': 300,
        'max_leaf_nodes': 63,
        'min_samples_leaf': 50,
        'l2_regularization': 0.0,
    }

class Model:
    def __init__(self, params=None):
        self.booster = None
        self.params = {} if params is None else dict(params)

    def fit(self, x, y, x_valid, y_valid):
        params = default_params()
        params.update(self.params)
        monotonic = None
        if x.shape[1] == 132:
            monotonic = np.zeros(x.shape[1], dtype=int)
            monotonic[16] = -1
        self.booster = HistGradientBoostingClassifier(
            learning_rate=params['learning_rate'],
            max_iter=params['max_iter'],
            max_leaf_nodes=params['max_leaf_nodes'],
            min_samples_leaf=params['min_samples_leaf'],
            l2_regularization=params['l2_regularization'],
            early_stopping=False,
            monotonic_cst=monotonic,
            random_state=7,
        )
        self.booster.fit(x, np.asarray(y).astype(int))

    def predict(self, x):
        return self.booster.predict_proba(x)[:, 1]

    def save(self, path):
        joblib.dump(self.booster, path + '.txt')

    def load(self, path):
        self.booster = joblib.load(path + '.txt')
