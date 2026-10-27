import numpy as np
import torch
from torch import nn

from sklearn.metrics import roc_auc_score

# 逐马二分类的两层网络。四个任务各训一份，含赔率 132 列，无赔率 131 列。
HIDDEN = (128, 64)
DROPOUT = 0.1
LR = 1e-3
USE_RESIDUAL = True
WEIGHT_DECAY = 2e-4
BATCH_SIZE = 4096
MAX_EPOCHS = 40
PATIENCE = 8
SEED = 7
GRAD_CLIP = 1.0

class Net(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        width_a, width_b = HIDDEN
        self.fc1 = nn.Linear(n_features, width_a)
        self.bn1 = nn.BatchNorm1d(width_a)
        self.fc2 = nn.Linear(width_a, width_b)
        self.bn2 = nn.BatchNorm1d(width_b)
        self.fc3 = nn.Linear(width_b, 1)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(DROPOUT)
        self.skip = nn.Linear(width_a, width_b) if USE_RESIDUAL else None

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop(x)
        h = x
        x = self.fc2(x)
        x = self.bn2(x)
        if self.skip is not None:
            x = x + self.skip(h)
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

    def predict(self, x):
        self.net.eval()
        x_scaled = self.transform(x)
        tensor = torch.tensor(x_scaled, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.net(tensor).view(-1)
            y_pred = torch.sigmoid(logits).detach().cpu().numpy()
            return y_pred

    def calculate_stats(self, y_pred, y_truth):
        y_truth = np.asarray(y_truth, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        
        # 1. AUC-ROC
        auc_roc = roc_auc_score(y_truth, y_pred)
        
        # 2. LogLoss / 交叉熵损失
        p_clip = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
        loss = -np.mean(y_truth * np.log(p_clip) + (1.0 - y_truth) * np.log(1.0 - p_clip))

        return {
            'auc': float(auc_roc),
            'logloss': float(loss),
        }


    def train(self, x, y, x_val, y_val):
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        y = np.asarray(y, dtype=np.float64).reshape(-1)
        y_val = np.asarray(y_val, dtype=np.float64).reshape(-1)

        self.n_features = int(np.asarray(x).shape[1])
        self.median, self.mean, self.std, x_train = self.fit_preprocess(x)

        self.net = Net(self.n_features).to(self.device)
        print(f'mlp device {self.device}')
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        
        x_tensor = torch.tensor(x_train, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        generator = torch.Generator()
        generator.manual_seed(SEED)
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_tensor, y_tensor),
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=generator,
        )

        best_auc = -1.0
        best_epoch = 0
        wait = 0
        best_state = None
        for epi in range(1, MAX_EPOCHS + 1):
            self.net.train()
            total = 0.0
            count = 0
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                loss = loss_fn(self.net(xb).view(-1), yb)
                loss.backward()
                if GRAD_CLIP is not None:
                    nn.utils.clip_grad_norm_(self.net.parameters(), GRAD_CLIP)
                optimizer.step()
                total += float(loss.item()) * len(yb)
                count += len(yb)

            y_val_pred = self.predict(x_val)
            val_stats = self.calculate_stats(y_val_pred, y_val)
            val_auc = val_stats['auc']
            print(f'Epoch {epi}: train loss {total / count:.4f}\t val loss {val_stats["logloss"]:.4f}\t val auc {val_auc:.4f}')

            if val_auc > best_auc:
                best_auc = val_auc
                best_epoch = epi
                wait = 0
                best_state = {key: value.detach().cpu().clone() for key, value in self.net.state_dict().items()}
            else:
                wait += 1
                if wait >= PATIENCE:
                    break
        self.net.load_state_dict(best_state)
        self.net.to(self.device)
        print(f'Best Epoch {best_epoch}: val auc {best_auc:.4f}')

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