from sklearn.linear_model import SGDClassifier

from models.ml.all.scaled import ImputedScaledModel

# SGD hinge，线性 SVM 的随机梯度版本。alpha=0.0001 是 sklearn 的默认 L2 强度。
# 没有概率。分数是 decision_function。
class Model(ImputedScaledModel):
    def estimator(self):
        return SGDClassifier(
            loss='hinge',
            penalty='l2',
            alpha=0.0001,
            max_iter=1000,
            random_state=7,
        )

    def score(self, x_scaled):
        return self.booster.decision_function(x_scaled)
