from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.ml.all.scaled import MedianFilledModel

# 软投票。基模型是 L2 逻辑回归、100 轮直方图 GBDT、100 棵极端随机树。
# 只使用 sklearn，避免把 LightGBM 和 XGBoost 装进同一个进程。
class Model(MedianFilledModel):
    def estimator(self):
        voters = VotingClassifier(
            estimators=[
                ('logistic', LogisticRegression(l1_ratio=0, C=1.0, solver='lbfgs', max_iter=500)),
                ('hist_gbdt', HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=100,
                    max_leaf_nodes=31,
                    min_samples_leaf=50,
                    early_stopping=False,
                    random_state=7,
                )),
                ('extra_trees', ExtraTreesClassifier(
                    n_estimators=100,
                    min_samples_leaf=50,
                    max_features=0.8,
                    random_state=7,
                    n_jobs=-1,
                )),
            ],
            voting='soft',
        )
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', voters),
        ])
