import joblib
import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier

# 可解释提升机。每列一条形状函数，interactions=0 不加交叉项。
# 132 列时第 16 列 win_odd 单调递减。缺失值交给 EBM 自己的缺失箱。
class Model:
    def __init__(self):
        self.booster = None

    def fit(self, x, y, x_valid, y_valid):
        monotone = None
        if x.shape[1] == 132:
            monotone = [0] * x.shape[1]
            monotone[16] = -1
        self.booster = ExplainableBoostingClassifier(
            interactions=0,
            outer_bags=4,
            max_rounds=100,
            learning_rate=0.05,
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
