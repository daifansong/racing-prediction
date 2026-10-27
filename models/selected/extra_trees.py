import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

# 极端随机树。切分点比随机森林更随机。缺失值直接传入。
def default_params():
    return {
        'n_estimators': 300,
        'max_depth': None,
        'min_samples_leaf': 50,
        'max_features': 0.8,
    }

class Model:
    def __init__(self, params=None):
        self.booster = None
        self.params = {} if params is None else dict(params)

    def fit(self, x, y, x_valid, y_valid):
        params = default_params()
        params.update(self.params)
        self.booster = ExtraTreesClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_leaf=params['min_samples_leaf'],
            max_features=params['max_features'],
            bootstrap=False,
            random_state=7,
            n_jobs=-1,
        )
        self.booster.fit(x, np.asarray(y).astype(int))

    def predict(self, x):
        return self.booster.predict_proba(x)[:, 1]

    def save(self, path):
        joblib.dump(self.booster, path + '.txt')

    def load(self, path):
        self.booster = joblib.load(path + '.txt')
