import joblib
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Bagging。基学习器接受缺失值。不传 params 时是 100 棵树，叶子至少 50 行。
def default_params():
    return {
        'n_estimators': 100,
        'max_samples': 1.0,
        'min_samples_leaf': 50,
        'max_depth': None,
    }

class Model:
    def __init__(self, params=None):
        self.booster = None
        self.params = {} if params is None else dict(params)

    def fit(self, x, y, x_valid, y_valid):
        params = default_params()
        params.update(self.params)
        self.booster = BaggingClassifier(
            estimator=DecisionTreeClassifier(
                min_samples_leaf=params['min_samples_leaf'],
                max_depth=params['max_depth'],
            ),
            n_estimators=params['n_estimators'],
            max_samples=params['max_samples'],
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
