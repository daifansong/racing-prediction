from sklearn.ensemble import GradientBoostingClassifier

from models.all.scaled import MedianFilledModel

# sklearn GBDT。不接受缺失值，沿用 MedianFilledModel 的训练集列中位数填充。
def default_params():
    return {
        'n_estimators': 100,
        'learning_rate': 0.05,
        'max_depth': 3,
        'min_samples_leaf': 1,
        'subsample': 0.8,
        'random_state': 7,
    }

class Model(MedianFilledModel):
    def __init__(self, params=None):
        super().__init__()
        self.params = {} if params is None else dict(params)

    def estimator(self):
        params = default_params()
        params.update(self.params)
        return GradientBoostingClassifier(**params)
