from catboost import CatBoostClassifier

# 保持 Lossguide，depth 固定 16。类别权重用 auto_class_weights，不设则与现在的 all 一致。
def default_params():
    return {
        'loss_function': 'Logloss',
        'eval_metric': 'Logloss',
        'learning_rate': 0.05,
        'grow_policy': 'Lossguide',
        'max_leaves': 63,
        'depth': 16,
        'min_data_in_leaf': 50,
        'l2_leaf_reg': 3.0,
        'rsm': 0.8,
        'bootstrap_type': 'Bernoulli',
        'subsample': 0.8,
        'random_seed': 7,
        'iterations': 1000,
        'early_stopping_rounds': 50,
        'use_best_model': True,
        'verbose': False,
        'allow_writing_files': False,
    }

class Model:
    def __init__(self, params=None):
        self.booster = None
        self.params = {} if params is None else dict(params)

    def fit(self, x, y, x_valid, y_valid):
        params = default_params()
        chosen = dict(self.params)
        if chosen.get('auto_class_weights') is None:
            chosen.pop('auto_class_weights', None)
        params.update(chosen)
        self.booster = CatBoostClassifier(**params)
        self.booster.fit(x, y, eval_set=(x_valid, y_valid))

    def predict(self, x):
        return self.booster.predict_proba(x)[:, 1]

    def save(self, path):
        self.booster.save_model(path + '.txt')

    def load(self, path):
        self.booster = CatBoostClassifier()
        self.booster.load_model(path + '.txt')
