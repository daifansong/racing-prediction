import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# 线性模型共用的缩放和存取。
# 距离大约一千多米，赔率是几倍到几十倍，胜率在 0 到 1。先按训练集标准化，正则、协方差和距离才不受列的单位影响。
# x_valid、y_valid 只为了和树模型的 fit 签名一致。这 8 个验证赛季不参与拟合，也不用来选正则。
class ScaledModel:
    def __init__(self):
        self.scaler = None
        self.booster = None

    def estimator(self):
        raise NotImplementedError

    def score(self, x_scaled):
        raise NotImplementedError

    def fit(self, x, y, x_valid, y_valid):
        y = np.asarray(y).astype(int)
        self.scaler = StandardScaler()
        x_scaled = self.scaler.fit_transform(x)
        self.booster = self.estimator()
        self.booster.fit(x_scaled, y)

    def predict(self, x):
        x_scaled = self.scaler.transform(x)
        return np.asarray(self.score(x_scaled), dtype=float).reshape(-1)

    def save(self, path):
        joblib.dump({'scaler': self.scaler, 'booster': self.booster}, path + '.txt')

    def load(self, path):
        saved = joblib.load(path + '.txt')
        self.scaler = saved['scaler']
        self.booster = saved['booster']


def column_medians(x):
    x = np.asarray(x, dtype=float)
    medians = np.nanmedian(x, axis=0)
    return np.where(np.isfinite(medians), medians, 0.0)


def fill_medians(x, medians):
    x = np.asarray(x, dtype=float)
    mask = ~np.isfinite(x)
    if not mask.any():
        return x
    filled = x.copy()
    filled[mask] = np.take(medians, np.nonzero(mask)[1])
    return filled


def stratified_sample(x, y, limit, seed=7):
    # 慢模型只拿一部分训练行。分层抽样保持这一任务的正样本比例，种子固定为 7。
    x = np.asarray(x)
    y = np.asarray(y).astype(int)
    if len(y) <= limit:
        return x, y
    rng = np.random.default_rng(seed)
    groups = []
    for label in (0, 1):
        index = np.flatnonzero(y == label)
        count = int(round(limit * (len(index) / len(y))))
        count = min(max(count, 1), len(index))
        groups.append(rng.choice(index, size=count, replace=False))
    chosen = np.concatenate(groups)
    if len(chosen) > limit:
        chosen = rng.choice(chosen, size=limit, replace=False)
    elif len(chosen) < limit:
        rest = np.setdiff1d(np.arange(len(y)), chosen)
        extra = rng.choice(rest, size=min(limit - len(chosen), len(rest)), replace=False)
        chosen = np.concatenate([chosen, extra])
    return x[chosen], y[chosen]


# 线性模型、SVM、近邻和朴素贝叶斯先按训练集的列中位数填缺失，再标准化。
# 树模型里不接受缺失值的估计器只用中位数，不做标准化。
class ImputedScaledModel(ScaledModel):
    def __init__(self):
        super().__init__()
        self.medians = None

    def _prepare_train(self, x):
        self.medians = column_medians(x)
        return fill_medians(x, self.medians)

    def _prepare(self, x):
        return fill_medians(x, self.medians)

    def fit(self, x, y, x_valid, y_valid):
        super().fit(self._prepare_train(x), y, x_valid, y_valid)

    def predict(self, x):
        return super().predict(self._prepare(x))

    def save(self, path):
        joblib.dump(
            {'scaler': self.scaler, 'booster': self.booster, 'medians': self.medians},
            path + '.txt',
        )

    def load(self, path):
        saved = joblib.load(path + '.txt')
        self.scaler = saved['scaler']
        self.booster = saved['booster']
        self.medians = saved['medians']


class MedianFilledModel:
    def __init__(self):
        self.medians = None
        self.booster = None

    def estimator(self):
        raise NotImplementedError

    def score(self, x_filled):
        return self.booster.predict_proba(x_filled)[:, 1]

    def fit(self, x, y, x_valid, y_valid):
        y = np.asarray(y).astype(int)
        self.medians = column_medians(x)
        self.booster = self.estimator()
        self.booster.fit(fill_medians(x, self.medians), y)

    def predict(self, x):
        return np.asarray(self.score(fill_medians(x, self.medians)), dtype=float).reshape(-1)

    def save(self, path):
        joblib.dump({'medians': self.medians, 'booster': self.booster}, path + '.txt')

    def load(self, path):
        saved = joblib.load(path + '.txt')
        self.medians = saved['medians']
        self.booster = saved['booster']
