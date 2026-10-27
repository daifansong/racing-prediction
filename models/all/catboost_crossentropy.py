from catboost import CatBoostClassifier

# CatBoost CrossEntropy。这个损失不能设置类别权重。
# Lossguide 树也不能加单调约束，树形状和 catboost.py 保持一致。
class Model:
    def __init__(self):
        self.booster = None

    def fit(self, x, y, x_valid, y_valid):
        params = dict(
            loss_function='CrossEntropy',
            eval_metric='Logloss',
            learning_rate=0.05,
            grow_policy='Lossguide',
            max_leaves=63,
            depth=16,
            min_data_in_leaf=50,
            rsm=0.8,
            bootstrap_type='Bernoulli',
            subsample=0.8,
            random_seed=7,
            iterations=1000,
            early_stopping_rounds=50,
            use_best_model=True,
            verbose=False,
            allow_writing_files=False,
        )
        self.booster = CatBoostClassifier(**params)
        self.booster.fit(x, y, eval_set=(x_valid, y_valid))

    def predict(self, x):
        return self.booster.predict_proba(x)[:, 1]

    def save(self, path):
        self.booster.save_model(path + '.txt')

    def load(self, path):
        self.booster = CatBoostClassifier()
        self.booster.load_model(path + '.txt')
