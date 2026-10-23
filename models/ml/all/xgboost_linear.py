import joblib
import numpy as np
import xgboost as xgb

from models.ml.all.scaled import column_medians, fill_medians

# XGBoost 线性提升。gblinear 不接受缺失值，先用训练集列中位数填充。
# 它不能把早停截到最佳轮，所以固定训练 200 轮。输出是正类概率。
class Model:
    def __init__(self):
        self.booster = None
        self.medians = None

    def fit(self, x, y, x_valid, y_valid):
        self.medians = column_medians(x)
        train_set = xgb.DMatrix(fill_medians(x, self.medians), label=y)
        valid_set = xgb.DMatrix(fill_medians(x_valid, self.medians), label=y_valid)
        self.booster = xgb.train(
            {
                'booster': 'gblinear',
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'learning_rate': 0.05,
                'alpha': 0.0,
                'lambda': 1.0,
                'seed': 7,
                'verbosity': 0,
            },
            train_set,
            num_boost_round=200,
            evals=[(valid_set, 'valid')],
            verbose_eval=False,
        )

    def predict(self, x):
        return self.booster.predict(xgb.DMatrix(fill_medians(x, self.medians)))

    def save(self, path):
        joblib.dump(
            {'raw': self.booster.save_raw(raw_format='json'), 'medians': self.medians},
            path + '.txt',
        )

    def load(self, path):
        saved = joblib.load(path + '.txt')
        self.medians = saved['medians']
        self.booster = xgb.Booster()
        self.booster.load_model(bytearray(saved['raw']))
