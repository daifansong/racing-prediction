from sklearn.linear_model import LogisticRegression

from models.all.scaled import ScaledModel

# L2 逻辑回归，这一组的线性基线。
# 23 万行、132 列用 lbfgs。l1_ratio=0 表示纯 L2；sklearn 1.8 起 penalty 将在 1.10 删除。
# C 固定为 1，不在 8 个验证赛季上调参。
# 分数是正类概率，越大越可能是独赢或前三。
class Model(ScaledModel):
    def estimator(self):
        return LogisticRegression(
            l1_ratio=0,
            C=1.0,
            solver='lbfgs',
            max_iter=500,
        )

    def score(self, x_scaled):
        return self.booster.predict_proba(x_scaled)[:, 1]
