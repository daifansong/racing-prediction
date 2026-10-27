from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from models.all.scaled import ScaledModel

# 二次判别分析。独赢类大约只有两万行，132 维的类协方差容易过拟合。
# reg_param=0.1 把每一类协方差向单位阵收 10%。这个值固定，不在 8 个验证赛季上挑选。
# 分数是正类概率，越大越可能是独赢或前三。
class Model(ScaledModel):
    def estimator(self):
        return QuadraticDiscriminantAnalysis(reg_param=0.1)

    def score(self, x_scaled):
        return self.booster.predict_proba(x_scaled)[:, 1]
