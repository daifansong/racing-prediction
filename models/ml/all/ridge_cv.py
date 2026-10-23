from sklearn.linear_model import RidgeClassifierCV

from models.ml.all.scaled import ImputedScaledModel

# RidgeClassifierCV。在 0.1、1、10 三个 alpha 里用广义交叉验证选择，不用外部验证集。
# 没有概率。分数是 decision_function，越大越靠近正类。
class Model(ImputedScaledModel):
    def estimator(self):
        return RidgeClassifierCV(alphas=(0.1, 1.0, 10.0))

    def score(self, x_scaled):
        return self.booster.decision_function(x_scaled)
