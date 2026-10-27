from sklearn.linear_model import SGDClassifier

from models.all.scaled import ImputedScaledModel

# 感知机。loss='perceptron'，不平均、不惩罚。错分才更新权重。
# 没有概率。分数是 decision_function。
class Model(ImputedScaledModel):
    def estimator(self):
        return SGDClassifier(
            loss='perceptron',
            penalty=None,
            max_iter=1000,
            average=False,
            random_state=7,
        )

    def score(self, x_scaled):
        return self.booster.decision_function(x_scaled)
