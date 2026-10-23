from sklearn.linear_model import SGDClassifier

from models.ml.all.scaled import ScaledModel

# 平均感知机。sklearn 1.9 的 Perceptron 没有 average，所以用 SGD 的感知机损失，并把每一轮系数取平均。
# penalty=None，不加 L2。顺序由 random_state=7 固定。
# 分数用 decision_function，越大越靠近正类，只用来在一场内排序。
class Model(ScaledModel):
    def estimator(self):
        return SGDClassifier(
            loss='perceptron',
            penalty=None,
            average=True,
            max_iter=1000,
            random_state=7,
        )

    def score(self, x_scaled):
        return self.booster.decision_function(x_scaled)
