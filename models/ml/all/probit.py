import joblib
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler

from models.ml.all.scaled import column_medians, fill_medians

# Probit 回归。连接函数是标准正态的分布函数，只保存系数，不把训练矩阵放进模型文件。
class Model:
    def __init__(self):
        self.medians = None
        self.scaler = None
        self.params = None

    def _design(self, x):
        filled = fill_medians(x, self.medians)
        return sm.add_constant(self.scaler.transform(filled), has_constant='add')

    def fit(self, x, y, x_valid, y_valid):
        self.medians = column_medians(x)
        self.scaler = StandardScaler()
        filled = self.scaler.fit_transform(fill_medians(x, self.medians))
        design = sm.add_constant(filled, has_constant='add')
        result = sm.Probit(np.asarray(y).astype(float), design).fit(disp=False, maxiter=100)
        self.params = np.asarray(result.params, dtype=float)

    def predict(self, x):
        eta = np.clip(self._design(x) @ self.params, -30, 30)
        return norm.cdf(eta)

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
