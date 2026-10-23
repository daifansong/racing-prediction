import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class Model:
    def __init__(self):
        self.booster = None

    def fit(self, x, y, x_valid, y_valid):
        self.booster = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=50,
            max_features=0.8,
            bootstrap=True,
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
