from sklearn.svm import LinearSVC

from models.ml.all.scaled import ScaledModel

# 线性支持向量机。样本远多于 132 列，dual=False，在原空间求解。C 固定为 1。
# 分数用 decision_function，越大越靠近正类，只用来在一场内排序。
class Model(ScaledModel):
    def estimator(self):
        return LinearSVC(
            C=1.0,
            dual=False,
            max_iter=5000,
            random_state=7,
        )

    def score(self, x_scaled):
        return self.booster.decision_function(x_scaled)
