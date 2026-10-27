from sklearn.svm import NuSVC

from models.all.scaled import ImputedScaledModel, stratified_sample

# RBF 核 NuSVC。nu=0.05，只用分层抽出的 2 万行。没有概率，分数是 decision_function。
class Model(ImputedScaledModel):
    def estimator(self):
        return NuSVC(kernel='rbf', nu=0.05, random_state=7)

    def score(self, x_scaled):
        return self.booster.decision_function(x_scaled)

    def fit(self, x, y, x_valid, y_valid):
        x, y = stratified_sample(x, y, 20000)
        super().fit(x, y, x_valid, y_valid)
