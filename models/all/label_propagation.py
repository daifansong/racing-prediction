from sklearn.semi_supervised import LabelPropagation

from models.all.scaled import ImputedScaledModel, stratified_sample

# 标签传播。knn 图在全量 20 万行上存不下，只用分层抽出的 2 万行。7 个近邻。
class Model(ImputedScaledModel):
    def estimator(self):
        return LabelPropagation(kernel='knn', n_neighbors=7)

    def score(self, x_scaled):
        return self.booster.predict_proba(x_scaled)[:, 1]

    def fit(self, x, y, x_valid, y_valid):
        x, y = stratified_sample(x, y, 20000)
        super().fit(x, y, x_valid, y_valid)
