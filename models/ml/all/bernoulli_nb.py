from sklearn.naive_bayes import BernoulliNB

from models.ml.all.scaled import ImputedScaledModel

# Bernoulli 朴素贝叶斯。先标准化，binarize=0 把高于该列均值的值记成 1。
class Model(ImputedScaledModel):
    def estimator(self):
        return BernoulliNB(binarize=0.0)

    def score(self, x_scaled):
        return self.booster.predict_proba(x_scaled)[:, 1]
