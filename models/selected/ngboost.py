from ngboost import NGBClassifier
from ngboost.distns import Bernoulli

from models.all.scaled import MedianFilledModel

# NGBoost。Bernoulli 分布做二分类，输出正类概率。不接受缺失值，先填训练集列中位数。
def default_params():
    return {
        'n_estimators': 100,
        'learning_rate': 0.05,
    }

class Model(MedianFilledModel):
    def __init__(self, params=None):
        super().__init__()
        self.params = {} if params is None else dict(params)

    def estimator(self):
        params = default_params()
        params.update(self.params)
        return NGBClassifier(
            Dist=Bernoulli,
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            verbose=False,
            random_state=7,
        )
