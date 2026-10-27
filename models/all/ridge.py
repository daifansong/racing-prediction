from sklearn.linear_model import RidgeClassifier

from models.all.scaled import ScaledModel

# 岭分类器。把 0/1 标签当成回归目标，用 L2 收缩，alpha=1。
# 它没有概率。分数用 decision_function，越大越靠近正类，只用来在一场内排序。
class Model(ScaledModel):
    def estimator(self):
        return RidgeClassifier(alpha=1.0)

    def score(self, x_scaled):
        return self.booster.decision_function(x_scaled)
