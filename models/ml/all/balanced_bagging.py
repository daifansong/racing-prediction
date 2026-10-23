from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier

from models.ml.all.scaled import MedianFilledModel

# 平衡 Bagging。100 个类别平衡的子集，基学习器是叶子至少 50 行的决策树。
# 采样器不接受缺失值，先用训练集的列中位数填充。
class Model(MedianFilledModel):
    def estimator(self):
        return BalancedBaggingClassifier(
            estimator=DecisionTreeClassifier(min_samples_leaf=50),
            n_estimators=100,
            random_state=7,
            n_jobs=-1,
        )
