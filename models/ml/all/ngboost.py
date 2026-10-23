from ngboost import NGBClassifier
from ngboost.distns import Bernoulli

from models.ml.all.scaled import MedianFilledModel

# NGBoost。用 Bernoulli 分布做二分类，100 轮，学习率 0.05。输出是正类概率。
# 不接受缺失值，先用训练集的列中位数填充。
class Model(MedianFilledModel):
    def estimator(self):
        return NGBClassifier(
            Dist=Bernoulli,
            n_estimators=100,
            learning_rate=0.05,
            verbose=False,
            random_state=7,
        )
