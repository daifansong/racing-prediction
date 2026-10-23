import joblib
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from models.ml.all.scaled import column_medians, fill_medians

# 随机傅里叶特征近似 RBF 核，再接 L2 逻辑回归。100 个分量，用全量训练集。
# 分数是正类概率。
class Model:
    def __init__(self):
        self.medians = None
        self.scaler = None
        self.sampler = None
        self.booster = None

    def fit(self, x, y, x_valid, y_valid):
        self.medians = column_medians(x)
        self.scaler = StandardScaler()
        x_scaled = self.scaler.fit_transform(fill_medians(x, self.medians))
        self.sampler = RBFSampler(gamma='scale', n_components=100, random_state=7)
        mapped = self.sampler.fit_transform(x_scaled)
        self.booster = LogisticRegression(l1_ratio=0, C=1.0, solver='lbfgs', max_iter=500)
        self.booster.fit(mapped, np.asarray(y).astype(int))

    def predict(self, x):
        x_scaled = self.scaler.transform(fill_medians(x, self.medians))
        return self.booster.predict_proba(self.sampler.transform(x_scaled))[:, 1]

    def save(self, path):
        joblib.dump(
            {
                'medians': self.medians,
                'scaler': self.scaler,
                'sampler': self.sampler,
                'booster': self.booster,
            },
            path + '.txt',
        )

    def load(self, path):
        saved = joblib.load(path + '.txt')
        self.medians = saved['medians']
        self.scaler = saved['scaler']
        self.sampler = saved['sampler']
        self.booster = saved['booster']
