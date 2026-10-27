import numpy as np
import xgboost as xgb

# 保持 lossguide。容量用 max_leaves，max_depth 仍是 0。
def default_params():
    return {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'learning_rate': 0.05,
        'grow_policy': 'lossguide',
        'max_leaves': 63,
        'max_depth': 0,
        'min_child_weight': 5,
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        'seed': 7,
        'verbosity': 0,
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
        train_set = xgb.DMatrix(x, label=y)
        valid_set = xgb.DMatrix(x_valid, label=y_valid)
        params = default_params()
        chosen = dict(self.params)
        multiplier = chosen.pop('pos_weight_mult', None)
        params.update(chosen)
        if multiplier is not None:
            params['scale_pos_weight'] = scale_pos_weight(y, multiplier)
        self.booster = xgb.train(
            params,
            train_set,
            num_boost_round=1000,
            evals=[(valid_set, 'valid')],
            verbose_eval=False,
            callbacks=[xgb.callback.EarlyStopping(rounds=50, save_best=True)],
        )

    def predict(self, x):
        return self.booster.predict(xgb.DMatrix(x))

    def save(self, path):
        raw = self.booster.save_raw(raw_format='json')
        with open(path + '.txt', 'wb') as model_file:
            model_file.write(raw)

    def load(self, path):
        self.booster = xgb.Booster()
        with open(path + '.txt', 'rb') as model_file:
            self.booster.load_model(bytearray(model_file.read()))
