import numpy as np
import torch
from torch import nn

from sklearn.metrics import roc_auc_score

# 逐马二分类的两层网络。四个任务各训一份，含赔率 132 列，无赔率 131 列。
# 训练集上做中位数填补和标准化。验证集只用来决定停在哪一轮，不参与这两步。
# 训练输出 logit，预测时做 sigmoid。分数是正类概率，越大越可能是独赢或前三。
HIDDEN = (64, 32)
DROPOUT = 0.1
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 4096
MAX_EPOCHS = 40
PATIENCE = 8
SEED = 7


class Net(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        width_a, width_b = HIDDEN
        self.fc1 = nn.Linear(n_features, width_a)
        self.fc2 = nn.Linear(width_a, width_b)
        self.fc3 = nn.Linear(width_b, 1)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc3(x)
        return x


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def fill_median(x, median):
    filled = np.array(x, dtype=np.float64, copy=True)
    missing = ~np.isfinite(filled)
    if missing.any():
        rows, cols = np.where(missing)
        filled[rows, cols] = median[cols]
    return filled


class Model:
    def __init__(self):
        self.net = None
        self.median = None
        self.mean = None
        self.std = None
        self.n_features = None
        self.device = pick_device()

    def transform(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError('\tscore saved failed')
        filled = fill_median(x, self.median)
        return (filled - self.mean) / self.std

    def fit_preprocess(self, x):
        x = np.asarray(x, dtype=np.float64)
        median = np.nanmedian(x, axis=0)
        median = np.where(np.isfinite(median), median, 0.0)
        filled = fill_median(x, median)
        mean = filled.mean(axis=0)
        std = filled.std(axis=0)
        std = np.where(std > 0, std, 1.0)
        return median, mean, std, (filled - mean) / std

    def scores(self, x_scaled):
        self.net.eval()
        tensor = torch.tensor(x_scaled, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.net(tensor).view(-1)
            return torch.sigmoid(logits).detach().cpu().numpy()

    def train(self, x, y, x_valid, y_valid):
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        y_valid = np.asarray(y_valid, dtype=np.float64).reshape(-1)
        self.n_features = int(np.asarray(x).shape[1])
        self.median, self.mean, self.std, x_train = self.fit_preprocess(x)
        x_valid = self.transform(x_valid)
        self.net = Net(self.n_features).to(self.device)
        print(f'mlp device {self.device}')
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        generator = torch.Generator()
        generator.manual_seed(SEED)
        x_tensor = torch.tensor(x_train, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_tensor, y_tensor),
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=generator,
        )
        best_auc = -1.0
        best_epoch = 0
        wait = 0
        best_state = None
        for epoch in range(1, MAX_EPOCHS + 1):
            self.net.train()
            total = 0.0
            count = 0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                loss = loss_fn(self.net(xb).view(-1), yb)
                loss.backward()
                optimizer.step()
                total += float(loss.item()) * len(yb)
                count += len(yb)
            val_auc = roc_auc_score(y_valid, self.scores(x_valid))
            print(f'epoch {epoch} train loss {total / count:.4f} val auc {val_auc:.4f}')
            if val_auc > best_auc:
                best_auc = val_auc
                best_epoch = epoch
                wait = 0
                best_state = {key: value.detach().cpu().clone() for key, value in self.net.state_dict().items()}
            else:
                wait += 1
                if wait >= PATIENCE:
                    break
        self.net.load_state_dict(best_state)
        self.net.to(self.device)
        print(f'best epoch {best_epoch} val auc {best_auc:.4f}')

    def predict(self, x):
        return np.asarray(self.scores(self.transform(x)), dtype=float).reshape(-1)

    def save(self, path):
        torch.save(
            {
                'state_dict': {key: value.detach().cpu() for key, value in self.net.state_dict().items()},
                'median': self.median,
                'mean': self.mean,
                'std': self.std,
                'n_features': self.n_features,
            },
            path + '.pt',
        )

    def load(self, path):
        saved = torch.load(path + '.pt', map_location='cpu', weights_only=False)
        self.median = np.asarray(saved['median'], dtype=np.float64)
        self.mean = np.asarray(saved['mean'], dtype=np.float64)
        self.std = np.asarray(saved['std'], dtype=np.float64)
        self.n_features = int(saved['n_features'])
        self.device = pick_device()
        self.net = Net(self.n_features)
        self.net.load_state_dict(saved['state_dict'])
        self.net.to(self.device)
