import joblib
import numpy as np
from pygam import LogisticGAM
from sklearn.preprocessing import StandardScaler

from models.ml.all.scaled import column_medians, fill_medians

# 逻辑 GAM。每列 8 个样条，lam=1，最多 30 轮。132 列的样条模型会比较慢。
class Model:
    def __init__(self):
        self.medians = None
        self.scaler = None
        self.booster = None

    def fit(self, x, y, x_valid, y_valid):
        self.medians = column_medians(x)
        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(fill_medians(x, self.medians))
        self.booster = LogisticGAM(n_splines=8, lam=1.0, max_iter=30)
        self.booster.fit(scaled, np.asarray(y).astype(int))

    def predict(self, x):
        scaled = self.scaler.transform(fill_medians(x, self.medians))
        return np.asarray(self.booster.predict_proba(scaled), dtype=float).reshape(-1)

    def save(self, path):
        joblib.dump(
            {'medians': self.medians, 'scaler': self.scaler, 'booster': self.booster},
            path + '.txt',
        )

    def load(self, path):
        saved = joblib.load(path + '.txt')
        self.medians = saved['medians']
        self.scaler = saved['scaler']
        self.booster = saved['booster']
