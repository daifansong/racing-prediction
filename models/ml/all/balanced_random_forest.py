import joblib
import numpy as np
from imblearn.ensemble import BalancedRandomForestClassifier

# 平衡随机森林。每棵树抽一个类别平衡的自助样本。300 棵，叶子至少 50 行。
# 这个估计器接受缺失值。
class Model:
    def __init__(self):
        self.booster = None

    def fit(self, x, y, x_valid, y_valid):
        self.booster = BalancedRandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=50,
            max_features=0.8,
            random_state=7,
            n_jobs=-1,
        )
        self.booster.fit(x, np.asarray(y).astype(int))

    def predict(self, x):
        return self.booster.predict_proba(x)[:, 1]

    def save(self, path):
        joblib.dump(self.booster, path + '.txt')

    def load(self, path):
        self.booster = joblib.load(path + '.txt')
