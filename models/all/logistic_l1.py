from sklearn.linear_model import LogisticRegression

from models.all.scaled import ScaledModel

# L1 逻辑回归。评分、胜率这些相关列里，大部分系数会被压成 0，只留下少数几列。
# l1_ratio=1 表示纯 L1，求解器用 saga。C 固定为 1。
# 分数是正类概率，越大越可能是独赢或前三。
class Model(ScaledModel):
    def estimator(self):
        return LogisticRegression(
            l1_ratio=1,
            C=1.0,
            solver='saga',
            max_iter=2000,
            random_state=7,
        )

    def score(self, x_scaled):
        return self.booster.predict_proba(x_scaled)[:, 1]
