import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from models.all.scaled import column_medians, fill_medians

# 贝叶斯逻辑回归的均值场变分推断。权重先验是标准正态，预测用后验均值。
# 30 个 epoch，批大小 4096，种子 7。不把整份训练矩阵存进模型文件。
class Model:
    def __init__(self):
        self.medians = None
        self.scaler = None
        self.weight = None
        self.bias = None

    def fit(self, x, y, x_valid, y_valid):
        self.medians = column_medians(x)
        self.scaler = StandardScaler()
        features = self.scaler.fit_transform(fill_medians(x, self.medians)).astype(np.float32)
        target = np.asarray(y, dtype=np.float32)
        generator = torch.Generator()
        generator.manual_seed(7)
        values = torch.as_tensor(features)
        labels = torch.as_tensor(target)
        row_count, column_count = values.shape
        mean = torch.zeros(column_count, requires_grad=True)
        rho = torch.full((column_count,), -2.0, requires_grad=True)
        bias = torch.zeros(1, requires_grad=True)
        optimizer = torch.optim.Adam([mean, rho, bias], lr=0.05)
        for _epoch in range(30):
            order = torch.randperm(row_count, generator=generator)
            for start in range(0, row_count, 4096):
                batch = order[start:start + 4096]
                std = torch.nn.functional.softplus(rho)
                weight = mean + std * torch.randn(column_count, generator=generator)
                eta = values[batch] @ weight + bias
                nll = torch.nn.functional.binary_cross_entropy_with_logits(eta, labels[batch])
                kl = 0.5 * torch.sum(mean ** 2 + std ** 2 - 1 - torch.log(std ** 2 + 1e-12))
                loss = nll + kl * (len(batch) / row_count)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.weight = mean.detach().numpy()
        self.bias = float(bias.detach())

    def predict(self, x):
        scaled = self.scaler.transform(fill_medians(x, self.medians))
        eta = np.clip(scaled @ self.weight + self.bias, -30, 30)
        return 1.0 / (1.0 + np.exp(-eta))

    def save(self, path):
        joblib.dump(
            {
                'medians': self.medians,
                'scaler': self.scaler,
                'weight': self.weight,
                'bias': self.bias,
            },
            path + '.txt',
        )

    def load(self, path):
        saved = joblib.load(path + '.txt')
        self.medians = saved['medians']
        self.scaler = saved['scaler']
        self.weight = saved['weight']
        self.bias = saved['bias']
