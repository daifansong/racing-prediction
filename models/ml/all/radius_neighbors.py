import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors, RadiusNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from models.ml.all.scaled import column_medians, fill_medians

# 半径近邻。半径取训练集到第 25 近邻距离的中位数，使多数行能找到邻居。
# 半径外的行用训练集里最多的类别，保证每一行都有有限概率。
class Model:
    def __init__(self):
        self.medians = None
        self.scaler = None
        self.booster = None

    def fit(self, x, y, x_valid, y_valid):
        y = np.asarray(y).astype(int)
        self.medians = column_medians(x)
        self.scaler = StandardScaler()
        x_scaled = self.scaler.fit_transform(fill_medians(x, self.medians))
        neighbor_count = min(25, len(y))
        neighbors = NearestNeighbors(n_neighbors=neighbor_count, algorithm='brute')
        neighbors.fit(x_scaled)
        distances, _index = neighbors.kneighbors(x_scaled)
        radius = float(np.median(distances[:, -1]))
        if radius <= 0:
            radius = 1.0
        self.booster = RadiusNeighborsClassifier(
            radius=radius,
            weights='distance',
            algorithm='brute',
            outlier_label='most_frequent',
            n_jobs=-1,
        )
        self.booster.fit(x_scaled, y)

    def predict(self, x):
        x_scaled = self.scaler.transform(fill_medians(x, self.medians))
        return self.booster.predict_proba(x_scaled)[:, 1]

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
