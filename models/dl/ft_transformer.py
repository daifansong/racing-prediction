import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from sklearn.metrics import roc_auc_score

# 逐马二分类 FT-Transformer：每个特征一个 token，在特征维上做注意力。
# 四个任务各训一份；含赔率 132 列，无赔率 131 列。
D_MODEL = 64    # 每个特征映射成的向量长度
N_HEADS = 4     # 多头注意力 几个头
N_LAYERS = 2    # Encoder layer 数量
FFN_MULT = 4    # 每层前馈网络中间层的放大倍数 64->256->64
DROPOUT = 0.1
LR = 5e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 1024
PRED_BATCH_SIZE = 2048
MAX_EPOCHS = 40
PATIENCE = 8
SEED = 7
GRAD_CLIP = 1.0


class FeatureTokenizer(nn.Module):
    """(B, d) -> (B, d, D)；每列一对 weight / bias，bias 同时承担列身份。"""

    def __init__(self, n_features: int, d_model: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias = nn.Parameter(torch.zeros(n_features, d_model))
        nn.init.normal_(self.weight, std=d_model ** -0.5)

    def forward(self, x):
        return x.unsqueeze(-1) * self.weight + self.bias


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha          # 正样本权重，None 表示不加权

    def forward(self, logits, targets):
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)      # 预测正确类别的概率
        loss = (1 - p_t) ** self.gamma * bce             # 容易分的样本权重变小
        if self.alpha is not None:
            a_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = a_t * loss
        return loss.mean()


class Net(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
    ) -> None:
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=FFN_MULT * d_model,
            dropout=DROPOUT,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        tokens = self.tokenizer(x)                       # (B, d, D)
        cls = self.cls.expand(x.size(0), -1, -1)         # (B, 1, D)
        seq = torch.cat([cls, tokens], dim=1)            # (B, d+1, D)
        out = self.encoder(seq)                          # (B, d+1, D)
        return self.head(out[:, 0, :])                   # (B, 1)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def fill_median(x, median):
    x_arr = np.asarray(x, dtype=np.float64)
    missing = ~np.isfinite(x_arr)
    if not missing.any():
        return x_arr
    filled_x = x_arr.copy()
    rows, cols = np.where(missing)
    filled_x[rows, cols] = median[cols]
    return filled_x


class Model:
    def __init__(self, feature_columns=None):
        self.feature_columns = None if feature_columns is None else list(feature_columns)
        self.net = None
        self.feature_median = None
        self.feature_mean = None
        self.feature_std = None
        self.n_features = None
        self.device = pick_device()

    def transform(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(
                f'expected shape (n, {self.n_features}), got {getattr(x, "shape", None)}'
            )
        filled_x = fill_median(x, self.feature_median)
        return (filled_x - self.feature_mean) / self.feature_std

    def fit_preprocess(self, x):
        x = np.asarray(x, dtype=np.float64)
        median = np.nanmedian(x, axis=0)
        median = np.where(np.isfinite(median), median, 0.0)
        filled_x = fill_median(x, median)
        mean = filled_x.mean(axis=0)
        std = filled_x.std(axis=0)
        std = np.where(std > 0, std, 1.0)
        return median, mean, std, (filled_x - mean) / std

    def predict(self, x):
        # 分批前向，避免整表一次过注意力爆显存
        self.net.eval()
        x_scaled = self.transform(x)
        outputs = []
        with torch.no_grad():
            for start in range(0, len(x_scaled), PRED_BATCH_SIZE):
                chunk = torch.tensor(
                    x_scaled[start:start + PRED_BATCH_SIZE],
                    dtype=torch.float32,
                    device=self.device,
                )
                logits = self.net(chunk).view(-1)
                outputs.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(outputs) if outputs else np.zeros(0)

    def calculate_stats(self, y_pred, y_truth):
        y_truth = np.asarray(y_truth, dtype=float).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
        if np.unique(y_truth).size < 2:
            auc_roc = float('nan')
        else:
            auc_roc = float(roc_auc_score(y_truth, y_pred))
        p_clip = np.clip(y_pred, 1e-15, 1.0 - 1e-15)    # 预测概率严格限制在 (0, 1) 开区间内
        loss = -np.mean(
            y_truth * np.log(p_clip) + (1.0 - y_truth) * np.log(1.0 - p_clip)
        )
        return {'auc': auc_roc, 'logloss': float(loss)}

    def train(self, x_train, y_train, x_val, y_val):
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
        y_val = np.asarray(y_val, dtype=np.float64).reshape(-1)

        self.n_features = int(np.asarray(x_train).shape[1])
        self.feature_median, self.feature_mean, self.feature_std, x_train = self.fit_preprocess(x_train)

        self.net = Net(self.n_features).to(self.device)
        print(
            f'ft-transformer device {self.device} d={self.n_features} '
            f'd_model={D_MODEL} heads={N_HEADS} layers={N_LAYERS}'
        )
        loss_fn = FocalLoss(gamma=2.0, alpha=None)
        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=MAX_EPOCHS
        )

        x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        generator = torch.Generator()
        generator.manual_seed(SEED)
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor),
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
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                loss = loss_fn(self.net(batch_x).view(-1), batch_y)
                loss.backward()
                if GRAD_CLIP is not None:
                    nn.utils.clip_grad_norm_(self.net.parameters(), GRAD_CLIP)
                optimizer.step()
                total += float(loss.item()) * len(batch_y)
                count += len(batch_y)
            scheduler.step()

            y_val_pred = self.predict(x_val)
            val_stats = self.calculate_stats(y_val_pred, y_val)
            val_auc = val_stats['auc']
            print(
                f'Epoch {epi}: train loss {total / count:.4f}\t'
                f'val loss {val_stats["logloss"]:.4f}\t'
                f'val auc {val_auc:.4f}\t'
                f'lr {scheduler.get_last_lr()[0]:.2e}'
            )

            if np.isnan(val_auc):
                val_auc = -val_stats['logloss']
            if val_auc > best_auc:
                best_auc = val_auc
                best_epoch = epi
                wait = 0
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.net.state_dict().items()
                }
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
                'state_dict': {
                    key: value.detach().cpu()
                    for key, value in self.net.state_dict().items()
                },
                'feature_median': self.feature_median,
                'feature_mean': self.feature_mean,
                'feature_std': self.feature_std,
                'n_features': self.n_features,
                'd_model': D_MODEL,
                'n_heads': N_HEADS,
                'n_layers': N_LAYERS,
            },
            path + '.pt',
        )

    def load(self, path):
        saved = torch.load(path + '.pt', map_location='cpu', weights_only=False)
        self.feature_median = np.asarray(
            saved.get('feature_median', saved.get('median')), dtype=np.float64
        )
        self.feature_mean = np.asarray(
            saved.get('feature_mean', saved.get('mean')), dtype=np.float64
        )
        self.feature_std = np.asarray(
            saved.get('feature_std', saved.get('std')), dtype=np.float64
        )
        self.n_features = int(saved['n_features'])
        self.device = pick_device()
        self.net = Net(
            self.n_features,
            d_model=int(saved.get('d_model', D_MODEL)),
            n_heads=int(saved.get('n_heads', N_HEADS)),
            n_layers=int(saved.get('n_layers', N_LAYERS)),
        )
        self.net.load_state_dict(saved['state_dict'])
        self.net.to(self.device)