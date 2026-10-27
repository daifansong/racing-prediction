from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from models.all.scaled import MedianFilledModel

# AdaBoost。100 轮，学习率 0.5，基学习器是深度 2 的决策树。sklearn 1.9 只有 SAMME。
# 这个估计器不接受缺失值，先用训练集的列中位数填充。
class Model(MedianFilledModel):
    def estimator(self):
        return AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2),
            n_estimators=100,
            learning_rate=0.5,
            random_state=7,
        )
