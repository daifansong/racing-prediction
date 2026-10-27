import xgboost as xgb

class Model:
    def __init__(self):
        self.booster = None

    def fit(self, x, y, x_valid, y_valid):
        train_set = xgb.DMatrix(x, label=y)
        valid_set = xgb.DMatrix(x_valid, label=y_valid)
        self.booster = xgb.train(
            {
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
            },
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
