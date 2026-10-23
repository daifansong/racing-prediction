from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold

from models.ml.all.scaled import ScaledModel

# L2 逻辑回归。正则强度在训练集内部用 5 折交叉验证选择，不用外面传入的 8 个验证赛季。
# l1_ratios=(0,) 把惩罚固定成 L2，只搜索 10 个 C。选择标准是负对数损失。
# 折是打乱后的分层抽样，random_state=7。
# 分数是正类概率，越大越可能是独赢或前三。
class Model(ScaledModel):
    def estimator(self):
        return LogisticRegressionCV(
            Cs=10,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=7),
            l1_ratios=(0,),
            solver='lbfgs',
            scoring='neg_log_loss',
            max_iter=500,
            random_state=7,
            n_jobs=-1,
            refit=True,
            use_legacy_attributes=False,
        )

    def score(self, x_scaled):
        return self.booster.predict_proba(x_scaled)[:, 1]
