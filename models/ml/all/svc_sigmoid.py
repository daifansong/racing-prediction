from sklearn.svm import SVC

from models.ml.all.scaled import ImputedScaledModel, stratified_sample

# Sigmoid 核 SVC。只用分层抽出的 2 万行。没有概率，分数是 decision_function。
class Model(ImputedScaledModel):
    def estimator(self):
        return SVC(kernel='sigmoid', random_state=7)

    def score(self, x_scaled):
        return self.booster.decision_function(x_scaled)

    def fit(self, x, y, x_valid, y_valid):
        x, y = stratified_sample(x, y, 20000)
        super().fit(x, y, x_valid, y_valid)
