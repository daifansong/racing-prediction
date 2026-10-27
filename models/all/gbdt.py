from sklearn.ensemble import GradientBoostingClassifier

from models.all.scaled import MedianFilledModel

# sklearn GBDT。100 棵、学习率 0.05、深度 3、子样本 0.8。
# 这个估计器不接受缺失值，先用训练集的列中位数填充。
class Model(MedianFilledModel):
    def estimator(self):
        return GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            random_state=7,
        )
