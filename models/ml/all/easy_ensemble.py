from imblearn.ensemble import EasyEnsembleClassifier

from models.ml.all.scaled import MedianFilledModel

# EasyEnsemble。10 个类别平衡的子集，每个子集上一组 AdaBoost。
# 采样器不接受缺失值，先用训练集的列中位数填充。
class Model(MedianFilledModel):
    def estimator(self):
        return EasyEnsembleClassifier(n_estimators=10, random_state=7, n_jobs=-1)
