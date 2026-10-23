import joblib
import numpy as np
from sklearn.naive_bayes import CategoricalNB

from models.ml.all.scaled import column_medians, fill_medians

# Categorical 朴素贝叶斯要求非负整数。每列按训练集分位数切成最多 10 箱。
# 常数列全部记为 0。
class Model:
    def __init__(self):
        self.medians = None
        self.edges = None
        self.booster = None

    def _codes(self, filled):
        codes = np.zeros(filled.shape, dtype=int)
        for j, bins in enumerate(self.edges):
            if bins is None:
                continue
            codes[:, j] = np.digitize(filled[:, j], bins)
        return codes

    def fit(self, x, y, x_valid, y_valid):
        self.medians = column_medians(x)
        filled = fill_medians(x, self.medians)
        self.edges = []
        for j in range(filled.shape[1]):
            quantiles = np.unique(np.quantile(filled[:, j], np.linspace(0, 1, 11)))
            self.edges.append(quantiles[1:-1] if len(quantiles) > 2 else None)
        self.booster = CategoricalNB()
        self.booster.fit(self._codes(filled), np.asarray(y).astype(int))

    def predict(self, x):
        return self.booster.predict_proba(self._codes(fill_medians(x, self.medians)))[:, 1]

    def save(self, path):
        joblib.dump(
            {'medians': self.medians, 'edges': self.edges, 'booster': self.booster},
            path + '.txt',
        )

    def load(self, path):
        saved = joblib.load(path + '.txt')
        self.medians = saved['medians']
        self.edges = saved['edges']
        self.booster = saved['booster']
