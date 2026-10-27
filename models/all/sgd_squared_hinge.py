from sklearn.linear_model import SGDClassifier

from models.all.scaled import ImputedScaledModel

# SGD squared hinge。惩罚间隔内的误差时用平方，而不是线性 hinge。
# 没有概率。分数是 decision_function。
class Model(ImputedScaledModel):
    def estimator(self):
        return SGDClassifier(
            loss='squared_hinge',
            penalty='l2',
            alpha=0.0001,
            max_iter=1000,
            random_state=7,
        )

    def score(self, x_scaled):
        return self.booster.decision_function(x_scaled)
