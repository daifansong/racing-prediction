from sklearn.linear_model import LogisticRegression

from models.all.scaled import ScaledModel

# Elastic Net，L1 和 L2 各占一半（l1_ratio=0.5）。
# 相关列不会像 L1 那样只留一个，也不会像 L2 那样全部保留。求解器用 saga，C 固定为 1。
# 分数是正类概率，越大越可能是独赢或前三。
class Model(ScaledModel):
    def estimator(self):
        return LogisticRegression(
            l1_ratio=0.5,
            C=1.0,
            solver='saga',
            max_iter=2000,
            random_state=7,
        )

    def score(self, x_scaled):
        return self.booster.predict_proba(x_scaled)[:, 1]
