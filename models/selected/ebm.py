import joblib
import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier

# 可解释提升机。interactions 固定为 0，不加交叉项。缺失值交给 EBM 自己的缺失箱。
def default_params():
    return {
        'outer_bags': 4,
        'max_rounds': 100,
        'learning_rate': 0.05,
    }

class Model:
    def __init__(self, params=None):
        self.booster = None
        self.params = {} if params is None else dict(params)

    def fit(self, x, y, x_valid, y_valid):
        params = default_params()
        params.update(self.params)
        monotone = None
        if x.shape[1] == 132:
            monotone = [0] * x.shape[1]
            monotone[16] = -1
        self.booster = ExplainableBoostingClassifier(
            interactions=0,
            outer_bags=params['outer_bags'],
            max_rounds=params['max_rounds'],
            learning_rate=params['learning_rate'],
            random_state=7,
            monotone_constraints=monotone,
        )
        self.booster.fit(x, np.asarray(y).astype(int))

    def predict(self, x):
        return self.booster.predict_proba(x)[:, 1]

    def save(self, path):
        joblib.dump(self.booster, path + '.txt')

    def load(self, path):
        self.booster = joblib.load(path + '.txt')
