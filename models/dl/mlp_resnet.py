import numpy as np
import torch
from torch import nn

from sklearn.metrics import roc_auc_score

WIDTH = 128
N_BLOCKS = 2
DROPOUT = 0.1
LR = 1e-3
WEIGHT_DECAY = 2e-4
BATCH_SIZE = 4096
MAX_EPOCHS = 40
PATIENCE = 8
SEED = 7
GRAD_CLIP = 1.0


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.bn1 = nn.BatchNorm1d(width)
        self.fc2 = nn.Linear(width, width)
        self.bn2 = nn.BatchNorm1d(width)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = x
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = x + h
        x = self.relu(x)
        return x


class Tower(nn.Module):
    def __init__(
        self, in_features: int, width: int, n_blocks: int, dropout: float
    ) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, width)
        self.bn = nn.BatchNorm1d(width)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [ResidualBlock(width, dropout) for _ in range(n_blocks)]
        )

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)
        for block_i in self.blocks:
            x = block_i(x)
        return x


class MultiTowerNet(nn.Module):
    """一塔时头是 Linear(width, 1)。多塔时先拼再过融合层。"""

    def __init__(
        self,
        tower_in_features: list[int],
        width: int,
        n_blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.towers = nn.ModuleList(
            [
                Tower(in_features, width, n_blocks, dropout)
                for in_features in tower_in_features
            ]
        )
        self.is_single_tower = len(tower_in_features) == 1
        if self.is_single_tower:
            self.head = nn.Linear(width, 1)
        else:
            self.fuse_fc = nn.Linear(width * len(tower_in_features), width)
            self.fuse_bn = nn.BatchNorm1d(width)
            self.fuse_relu = nn.ReLU()
            self.fuse_drop = nn.Dropout(dropout)
            self.head = nn.Linear(width, 1)

    def forward(self, *xs):
        parts = []
        for tower, x in zip(self.towers, xs):
            parts.append(tower(x))
        if self.is_single_tower:
            x = parts[0]
        else:
            x = torch.cat(parts, dim=1)
            x = self.fuse_fc(x)
            x = self.fuse_bn(x)
            x = self.fuse_relu(x)
            x = self.fuse_drop(x)
            
        x = self.head(x)
        return x


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


def resolve_column_indices(feature_columns, layout):
    feature_name_to_idx = {name: i for i, name in enumerate(feature_columns)}
    tower_indices = []
    assigned_features = set()
    for group_i in layout:
        idx = []
        for feature_name in group_i:
            if feature_name not in feature_name_to_idx:
                continue
            if feature_name in assigned_features:
                raise ValueError(f'column assigned to more than one tower: {feature_name}')
            assigned_features.add(feature_name)
            idx.append(feature_name_to_idx[feature_name])
        if not idx:
            raise ValueError('one of the towers got an empty column set')
        tower_indices.append(np.asarray(idx, dtype=np.int64))
    missing = set(feature_columns) - assigned_features
    if missing:
        raise ValueError(f'columns not assigned to any tower: {sorted(missing)}')
    return tower_indices


class ResNetModel:
    layout = None
    log_name = 'mlp_resnet'

    def __init__(self, feature_columns=None):
        self.feature_columns = None if feature_columns is None else list(feature_columns)
        self.tower_indices = None
        if self.layout is not None:
            if self.feature_columns is None:
                raise ValueError('feature_columns is required')
            self.tower_indices = resolve_column_indices(self.feature_columns, self.layout)
        self.net = None
        self.feature_median = None
        self.feature_mean = None
        self.feature_std = None
        self.n_features = None
        self.device = pick_device()

    def _split_by_towers(self, x_scaled):
        if self.tower_indices is None:
            return (x_scaled,)
        return tuple(x_scaled[:, idx] for idx in self.tower_indices)

    def _make_net(self):
        n_inputs = [len(idx) for idx in self.tower_indices]
        return MultiTowerNet(n_inputs, WIDTH, N_BLOCKS, DROPOUT).to(self.device)

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
        self.net.eval()
        tower_parts = self._split_by_towers(self.transform(x))
        towers_tensors = [
            torch.tensor(tower, dtype=torch.float32, device=self.device)
            for tower in tower_parts
        ]
        with torch.no_grad():
            logits = self.net(*towers_tensors).view(-1)
            return torch.sigmoid(logits).detach().cpu().numpy()

    def calculate_stats(self, y_pred, y_truth):
        y_truth = np.asarray(y_truth, dtype=float).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
        if np.unique(y_truth).size < 2:
            auc_roc = float('nan')
        else:
            auc_roc = float(roc_auc_score(y_truth, y_pred))
        p_clip = np.clip(y_pred, 1e-15, 1.0 - 1e-15)    # 预测概率严格限制在了 (0, 1) 的开区间内，绝不允许预测概率精确等于 0 或 1
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
        if self.feature_columns is not None and self.n_features != len(self.feature_columns):
            raise ValueError(
                f'n_features {self.n_features} != len(feature_columns) '
                f'{len(self.feature_columns)}'
            )
        if self.tower_indices is None:
            self.tower_indices = [np.arange(self.n_features, dtype=np.int64)]

        self.feature_median, self.feature_mean, self.feature_std, x_train = self.fit_preprocess(x_train)
        tower_parts = self._split_by_towers(x_train)
        self.net = self._make_net()
        if self.layout is None: # 单塔模式
            print(f'{self.log_name} device {self.device} width {WIDTH} blocks {N_BLOCKS}')
        else:   # 多塔模式
            sizes = ' '.join(str(len(idx)) for idx in self.tower_indices)
            print(
                f'{self.log_name} device {self.device} width {WIDTH} '
                f'blocks {N_BLOCKS} columns {sizes}'
            )

        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )
        towers_tensors = [
            torch.tensor(tower, dtype=torch.float32)
            for tower in tower_parts
        ]
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

        generator = torch.Generator()
        generator.manual_seed(SEED)
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(*towers_tensors, y_train_tensor),
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
            for *batch_towers_x, batch_y in train_loader:
                batch_towers_x = [item.to(self.device) for item in batch_towers_x]
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                loss = loss_fn(self.net(*batch_towers_x).view(-1), batch_y)
                loss.backward()
                if GRAD_CLIP is not None:
                    nn.utils.clip_grad_norm_(self.net.parameters(), GRAD_CLIP)
                optimizer.step()
                total += float(loss.item()) * len(batch_y)
                count += len(batch_y)

            y_val_pred = self.predict(x_val)
            val_stats = self.calculate_stats(y_val_pred, y_val)
            val_auc = val_stats['auc']
            print(
                f'Epoch {epi}: train loss {total / count:.4f}\t'
                f'val loss {val_stats["logloss"]:.4f}\t'
                f'val auc {val_auc:.4f}'
            )
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
                'feature_columns': self.feature_columns,
                'tower_indices': self.tower_indices,
                'width': WIDTH,
                'n_blocks': N_BLOCKS,
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
        self.feature_columns = saved.get('feature_columns')
        tower_indices = saved.get('tower_indices') or saved.get('column_indices')
        if tower_indices is None:
            self.tower_indices = [np.arange(self.n_features, dtype=np.int64)]
        else:
            self.tower_indices = [
                np.asarray(idx, dtype=np.int64) for idx in tower_indices
            ]
        self.device = pick_device()
        self.net = self._make_net()
        self.net.load_state_dict(saved['state_dict'])
        self.net.to(self.device)
