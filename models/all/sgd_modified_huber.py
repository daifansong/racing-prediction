from sklearn.linear_model import SGDClassifier

from models.all.scaled import ScaledModel

# SGD，损失是 modified_huber。对特别大的赔率和评分比对数损失更稳。
# 正则仍是 L2，alpha=0.0001。不开启 early_stopping，8 个验证赛季不参与训练。
# 分数是正类概率，越大越可能是独赢或前三。
class Model(ScaledModel):
    def estimator(self):
        return SGDClassifier(
            loss='modified_huber',
            penalty='l2',
            alpha=0.0001,
            max_iter=1000,
            random_state=7,
        )

    def score(self, x_scaled):
        return self.booster.predict_proba(x_scaled)[:, 1]
