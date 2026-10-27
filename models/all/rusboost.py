from imblearn.ensemble import RUSBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from models.all.scaled import MedianFilledModel

# RUSBoost。每轮先随机欠采样到类别平衡，再做 50 轮提升，学习率 0.5。
# 基学习器是深度 2 的树。采样器不接受缺失值，先用训练集的列中位数填充。
class Model(MedianFilledModel):
    def estimator(self):
        return RUSBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2),
            n_estimators=50,
            learning_rate=0.5,
            random_state=7,
        )
