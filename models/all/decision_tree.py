import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# 单棵决策树。深度 8，叶子至少 50 行，避免在 20 万行上长满。缺失值直接传入。
class Model:
    def __init__(self):
        self.booster = None

    def fit(self, x, y, x_valid, y_valid):
        self.booster = DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=50,
            random_state=7,
        )
        self.booster.fit(x, np.asarray(y).astype(int))

    def predict(self, x):
        return self.booster.predict_proba(x)[:, 1]

    def save(self, path):
        joblib.dump(self.booster, path + '.txt')

    def load(self, path):
        self.booster = joblib.load(path + '.txt')
