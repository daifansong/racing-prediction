import joblib
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler

from models.all.scaled import column_medians, fill_medians

# Multinomial 朴素贝叶斯只接受非负特征。先填缺失、标准化，再减去训练集每列的最小值。
class Model:
    def __init__(self):
        self.medians = None
        self.scaler = None
        self.col_min = None
        self.booster = None

    def _shift(self, x_scaled):
        return np.maximum(x_scaled - self.col_min, 0.0)

    def fit(self, x, y, x_valid, y_valid):
        self.medians = column_medians(x)
        self.scaler = StandardScaler()
        x_scaled = self.scaler.fit_transform(fill_medians(x, self.medians))
        self.col_min = x_scaled.min(axis=0)
        self.booster = MultinomialNB()
        self.booster.fit(self._shift(x_scaled), np.asarray(y).astype(int))

    def predict(self, x):
        x_scaled = self.scaler.transform(fill_medians(x, self.medians))
        return self.booster.predict_proba(self._shift(x_scaled))[:, 1]

    def save(self, path):
        joblib.dump(
            {
                'medians': self.medians,
                'scaler': self.scaler,
                'col_min': self.col_min,
                'booster': self.booster,
            },
            path + '.txt',
        )

    def load(self, path):
        saved = joblib.load(path + '.txt')
        self.medians = saved['medians']
        self.scaler = saved['scaler']
        self.col_min = saved['col_min']
        self.booster = saved['booster']
