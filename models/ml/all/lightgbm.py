import lightgbm as lgb

class Model:
    def __init__(self):
        self.booster = None

    def fit(self, x, y, x_valid, y_valid):
        train_set = lgb.Dataset(x, label=y)
        valid_set = lgb.Dataset(x_valid, label=y_valid, reference=train_set)
        params = {
            'objective': 'binary',
            'learning_rate': 0.05,
            'num_leaves': 63,
            'min_child_samples': 50,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'seed': 7,
            'verbosity': -1,
        }
        # Column 16 is win_odd. The 131-column tasks have already dropped it.
        if x.shape[1] == 132:
            monotone = [0] * x.shape[1]
            monotone[16] = -1
            params['monotone_constraints'] = monotone
        self.booster = lgb.train(
            params,
            train_set,
            num_boost_round=1000,
            valid_sets=[valid_set],
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(0),
            ],
        )

    def predict(self, x):
        return self.booster.predict(x)

    def save(self, path):
        self.booster.save_model(path + '.txt')

    def load(self, path):
        self.booster = lgb.Booster(model_file=path + '.txt')