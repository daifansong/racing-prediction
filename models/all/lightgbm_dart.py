import numpy as np
import lightgbm as lgb

# LightGBM DART。Dropout 加在提升树上，这个模式没有早停，固定训练 1000 轮。
# scale_pos_weight 用训练集的负样本数除以正样本数。132 列时第 16 列 win_odd 单调递减。
class Model:
    def __init__(self):
        self.booster = None

    def fit(self, x, y, x_valid, y_valid):
        y = np.asarray(y)
        n_pos = float(np.sum(y == 1))
        n_neg = float(len(y) - n_pos)
        train_set = lgb.Dataset(x, label=y)
        params = {
            'objective': 'binary',
            'boosting_type': 'dart',
            'learning_rate': 0.05,
            'num_leaves': 63,
            'min_child_samples': 50,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'scale_pos_weight': n_neg / n_pos if n_pos else 1.0,
            'seed': 7,
            'verbosity': -1,
        }
        if x.shape[1] == 132:
            monotone = [0] * x.shape[1]
            monotone[16] = -1
            params['monotone_constraints'] = monotone
        self.booster = lgb.train(
            params,
            train_set,
            num_boost_round=1000,
            valid_sets=[lgb.Dataset(x_valid, label=y_valid, reference=train_set)],
            callbacks=[lgb.log_evaluation(0)],
        )

    def predict(self, x):
        return self.booster.predict(x)

    def save(self, path):
        self.booster.save_model(path + '.txt')

    def load(self, path):
        self.booster = lgb.Booster(model_file=path + '.txt')
