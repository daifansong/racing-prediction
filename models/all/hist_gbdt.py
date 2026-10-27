import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

# sklearn 的直方图 GBDT。固定 300 轮，不用它内部划出的验证集做早停。
# 缺失值直接传入。132 列时第 16 列 win_odd 单调递减。
class Model:
    def __init__(self):
        self.booster = None

    def fit(self, x, y, x_valid, y_valid):
        monotonic = None
        if x.shape[1] == 132:
            monotonic = np.zeros(x.shape[1], dtype=int)
            monotonic[16] = -1
        self.booster = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=63,
            min_samples_leaf=50,
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
