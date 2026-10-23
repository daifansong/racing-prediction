from sklearn.linear_model import SGDClassifier

from models.ml.all.scaled import ScaledModel

# 用 SGD 训练的 L2 逻辑回归，损失是 log_loss，学到的仍是逻辑回归。
# alpha=0.0001 是 sklearn 的默认 L2 强度。不从训练集再切验证，也不用 8 个验证赛季早停。
# 分数是正类概率，越大越可能是独赢或前三。
class Model(ScaledModel):
    def estimator(self):
        return SGDClassifier(
            loss='log_loss',
            penalty='l2',
            alpha=0.0001,
            max_iter=1000,
            random_state=7,
        )

    def score(self, x_scaled):
        return self.booster.predict_proba(x_scaled)[:, 1]
