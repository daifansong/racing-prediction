from sklearn.semi_supervised import LabelSpreading

from models.ml.all.scaled import ImputedScaledModel, stratified_sample

# 标签扩散。在标签传播上加了夹紧系数 alpha=0.2。只用分层抽出的 2 万行。
class Model(ImputedScaledModel):
    def estimator(self):
        return LabelSpreading(kernel='knn', n_neighbors=7, alpha=0.2)

    def score(self, x_scaled):
        return self.booster.predict_proba(x_scaled)[:, 1]

    def fit(self, x, y, x_valid, y_valid):
        x, y = stratified_sample(x, y, 20000)
        super().fit(x, y, x_valid, y_valid)
