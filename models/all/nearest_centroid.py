import numpy as np
from sklearn.neighbors import NearestCentroid

from models.all.scaled import ScaledModel

# 最近质心。每一类只有一个中心，只适合做很弱的基线。
# 分数是到 0 类中心的距离减去到 1 类中心的距离，越靠近正类越高，只用来在一场内排序。
class Model(ScaledModel):
    def estimator(self):
        return NearestCentroid()

    def score(self, x_scaled):
        # centroids_ 按 classes_ 排列。标签是 0/1，第 0 列是负类，第 1 列是正类。
        distances = np.linalg.norm(x_scaled[:, None, :] - self.booster.centroids_[None, :, :], axis=2)
        return distances[:, 0] - distances[:, 1]
