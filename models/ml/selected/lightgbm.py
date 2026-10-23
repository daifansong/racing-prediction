import lightgbm as lgb
import numpy as np

# 当前 all 里的默认参数。不传 params 时和那边一致，不设 scale_pos_weight。
def default_params():
    return {
        'objective': 'binary',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'min_child_samples': 50,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'seed': 7,
        'verbosity': -1,
        'early_stopping_rounds': 50,
        'num_boost_round': 1000,
    }

# pos_weight_mult 由配置传入，在 fit 里乘上该任务训练集的负正比。
def scale_pos_weight(y, multiplier):
    y = np.asarray(y)
    n_pos = float(np.sum(y == 1))
    n_neg = float(np.sum(y == 0))
    return n_neg / n_pos * float(multiplier)

class Model:
    def __init__(self, params=None):
        self.booster = None
        self.params = {} if params is None else dict(params)

    def fit(self, x, y, x_valid, y_valid):
        train_set = lgb.Dataset(x, label=y)
        valid_set = lgb.Dataset(x_valid, label=y_valid, reference=train_set)
        params = default_params()
        chosen = dict(self.params)
        multiplier = chosen.pop('pos_weight_mult', None)
        params.update(chosen)
        # 这两项只控制训练轮数，不能留在 lgb.train 的参数里。
        stopping_rounds = int(params.pop('early_stopping_rounds'))
        boost_rounds = int(params.pop('num_boost_round'))
        if multiplier is not None:
            params['scale_pos_weight'] = scale_pos_weight(y, multiplier)
        # Column 16 is win_odd. The 131-column tasks have already dropped it.
        if x.shape[1] == 132:
            monotone = [0] * x.shape[1]
            monotone[16] = -1
            params['monotone_constraints'] = monotone
        self.booster = lgb.train(
            params,
            train_set,
            num_boost_round=boost_rounds,
            valid_sets=[valid_set],
            callbacks=[
                lgb.early_stopping(stopping_rounds),
                lgb.log_evaluation(0),
            ],
        )

    def predict(self, x):
        return self.booster.predict(x)

    def save(self, path):
        self.booster.save_model(path + '.txt')

    def load(self, path):
        self.booster = lgb.Booster(model_file=path + '.txt')
