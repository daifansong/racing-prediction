from sklearn.neighbors import KNeighborsClassifier

from models.ml.all.scaled import ImputedScaledModel

# 25 个近邻，距离越近权重越大。brute 在标准化后的 132 维上比树索引稳。
# 预测要扫过全部训练行，会比树模型慢。
class Model(ImputedScaledModel):
    def estimator(self):
        return KNeighborsClassifier(
            n_neighbors=25,
            weights='distance',
            algorithm='brute',
            n_jobs=-1,
        )

    def score(self, x_scaled):
        return self.booster.predict_proba(x_scaled)[:, 1]
