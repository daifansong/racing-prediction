from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.ml.all.scaled import MedianFilledModel

# 对 L2 逻辑回归做 isotonic 校准。折分在训练集内部，3 折打乱分层，种子 7。
# 标准化在全部训练行上拟合，然后再进入这 3 折。外部验证集不参与。
class Model(MedianFilledModel):
    def estimator(self):
        logistic = LogisticRegression(l1_ratio=0, C=1.0, solver='lbfgs', max_iter=500)
        calibrated = CalibratedClassifierCV(
            estimator=logistic,
            method='isotonic',
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=7),
        )
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', calibrated),
        ])
