from sklearn.naive_bayes import GaussianNB

from models.ml.all.scaled import ImputedScaledModel

# 高斯朴素贝叶斯。假设标准化后的各列近似独立高斯。var_smoothing=1e-9 避免方差为 0。
class Model(ImputedScaledModel):
    def estimator(self):
        return GaussianNB(var_smoothing=1e-9)

    def score(self, x_scaled):
        return self.booster.predict_proba(x_scaled)[:, 1]
