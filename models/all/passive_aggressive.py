from sklearn.linear_model import SGDClassifier

from models.all.scaled import ScaledModel

# Passive-Aggressive（PA-I）。只有分错的样本才更新系数，结果受样本顺序影响。
# sklearn 1.8 起 PassiveAggressiveClassifier 将在 1.10 删除，这里用官方给出的等价写法。
# learning_rate='pa1'，eta0=1 就是原来的 C=1。顺序由 random_state=7 固定。
# 分数用 decision_function，越大越靠近正类，只用来在一场内排序。
class Model(ScaledModel):
    def estimator(self):
        return SGDClassifier(
            loss='hinge',
            penalty=None,
            learning_rate='pa1',
            eta0=1.0,
            max_iter=1000,
            random_state=7,
        )

    def score(self, x_scaled):
        return self.booster.decision_function(x_scaled)
