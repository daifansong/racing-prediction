from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from models.ml.all.scaled import ScaledModel

# 收缩线性判别分析，作为生成式线性基线。
# solver='lsqr'，shrinkage='auto' 用 Ledoit-Wolf 收缩协方差。
# 分数是正类概率，越大越可能是独赢或前三。
class Model(ScaledModel):
    def estimator(self):
        return LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')

    def score(self, x_scaled):
        return self.booster.predict_proba(x_scaled)[:, 1]
