import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from models.ml.all.scaled import column_medians, fill_medians

# Firth 逻辑回归。当前环境没有 Python 3.12 的 firthlogist，这里直接做偏差校正的迭代加权最小二乘。
# 每轮把帽子矩阵对角线 h 加进得分：y - mu + h * (0.5 - mu)。只保存系数。
class Model:
    def __init__(self):
        self.medians = None
        self.scaler = None
        self.params = None

    def _design(self, x):
        filled = fill_medians(x, self.medians)
        scaled = self.scaler.transform(filled)
        return np.column_stack([np.ones(len(scaled)), scaled])

    def fit(self, x, y, x_valid, y_valid):
        y = np.asarray(y, dtype=float)
        self.medians = column_medians(x)
        self.scaler = StandardScaler()
        self.scaler.fit(fill_medians(x, self.medians))
        design = self._design(x)
        beta = np.zeros(design.shape[1])
        for _round in range(50):
            eta = np.clip(design @ beta, -30, 30)
            mu = 1.0 / (1.0 + np.exp(-eta))
            weight = np.maximum(mu * (1.0 - mu), 1e-12)
            weighted = design * weight[:, None]
            gram = design.T @ weighted
            covariance = np.linalg.pinv(gram)
            hat = np.sum((design @ covariance) * weighted, axis=1)
            score = design.T @ (y - mu + hat * (0.5 - mu))
            step = covariance @ score
            beta = beta + step
            if np.max(np.abs(step)) < 1e-6:
                break
        self.params = beta

    def predict(self, x):
        eta = np.clip(self._design(x) @ self.params, -30, 30)
        return 1.0 / (1.0 + np.exp(-eta))

    def save(self, path):
        joblib.dump(
            {'medians': self.medians, 'scaler': self.scaler, 'params': self.params},
            path + '.txt',
        )

    def load(self, path):
        saved = joblib.load(path + '.txt')
        self.medians = saved['medians']
        self.scaler = saved['scaler']
        self.params = saved['params']
