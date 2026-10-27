import numpy as np
import xgboost as xgb

# XGBoost DART。booster 换成 dart，其余和默认的 gbtree 逻辑回归相同。
# scale_pos_weight 用训练集的负样本数除以正样本数。132 列时第 16 列 win_odd 单调递减。
class Model:
    def __init__(self):
        self.booster = None

    def fit(self, x, y, x_valid, y_valid):
        y = np.asarray(y)
        n_pos = float(np.sum(y == 1))
        n_neg = float(len(y) - n_pos)
        params = {
            'booster': 'dart',
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'learning_rate': 0.05,
            'grow_policy': 'lossguide',
            'max_leaves': 63,
            'max_depth': 0,
            'min_child_weight': 5,
            'colsample_bytree': 0.8,
            'subsample': 0.8,
            'scale_pos_weight': n_neg / n_pos if n_pos else 1.0,
            'seed': 7,
            'verbosity': 0,
        }
        if x.shape[1] == 132:
            params['monotone_constraints'] = (0,) * 16 + (-1,) + (0,) * (x.shape[1] - 17)
        train_set = xgb.DMatrix(x, label=y)
        valid_set = xgb.DMatrix(x_valid, label=y_valid)
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
