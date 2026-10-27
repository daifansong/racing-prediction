import numpy as np
from sklearn.linear_model import LogisticRegression

from models.all.scaled import ImputedScaledModel

# 无正则逻辑回归。C=np.inf 去掉惩罚；sklearn 1.8 起用这个写法，不再传 penalty=None。
# 缺失值用训练集列中位数填充，再标准化。分数是正类概率。
class Model(ImputedScaledModel):
    def estimator(self):
        return LogisticRegression(C=np.inf, solver='lbfgs', max_iter=500)

    def score(self, x_scaled):
        return self.booster.predict_proba(x_scaled)[:, 1]
