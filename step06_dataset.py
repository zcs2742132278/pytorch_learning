"""
======================================================================
第 6 步：Dataset 重构 — 自定义数据集
======================================================================
学习目标：
  1. 理解 torch.utils.data.Dataset 的设计思想
  2. 掌握 __len__ 和 __getitem__ 的实现
  3. 将第 4 步的分类代码解耦为 Dataset + Model + Train 三层
  4. 理解为什么要把数据加载从训练逻辑中分离

关键词：Dataset, __getitem__, __len__, 数据解耦
======================================================================

为什么需要 Dataset？
  - 数据加载逻辑与训练逻辑解耦
  - 代码复用：同一个 Dataset 可被不同模型使用
  - 配合 DataLoader 实现批量加载、打乱、多进程等
  - PyTorch 生态的通用接口（transform、collate 等）
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from utils import set_seed

set_seed(42)

# ============================================================
print("=" * 60)
print("6.1 自定义 Dataset 的基本结构")
print("=" * 60)


class MyDataset(Dataset):
    """自定义数据集 — 最小实现"""
    def __init__(self, features, labels):
        """
        Args:
            features: 特征数据 (N, D) numpy array 或 tensor
            labels:   标签数据 (N,)  numpy array 或 tensor
        """
        # 在 __init__ 中完成数据预处理（只需一次）
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        """返回数据集大小（必须实现）"""
        return len(self.features)

    def __getitem__(self, idx):
        """返回单个样本（必须实现）"""
        # 可在此做在线数据增强、归一化等
        return self.features[idx], self.labels[idx]


# 使用示例
x, y = make_blobs(n_samples=500, n_features=2, centers=3, random_state=42)
dataset = MyDataset(x, y)

print(f"数据集大小: {len(dataset)}")
print(f"dataset[0] = {dataset[0]}  # 返回 (特征, 标签) 元组")
print(f"  feature shape: {dataset[0][0].shape}")
print(f"  label: {dataset[0][1]}")
print()

# ============================================================
print("=" * 60)
print("6.2 带预处理的数据集")
print("=" * 60)


class NormalizedDataset(Dataset):
    """在 __init__ 中计算统计量并做归一化"""
    def __init__(self, features, labels, normalize=True):
        self.labels = torch.LongTensor(labels)

        # 转 float
        features = torch.FloatTensor(features)

        # 计算归一化参数（只计算一次）
        if normalize:
            self.mean = features.mean(dim=0, keepdim=True)
            self.std = features.std(dim=0, keepdim=True)
            self.features = (features - self.mean) / (self.std + 1e-8)
        else:
            self.features = features
            self.mean = None
            self.std = None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def denormalize(self, features):
        """将归一化后的特征还原"""
        if self.mean is not None:
            return features * self.std + self.mean
        return features


# 对比有无归一化
x, y = make_blobs(n_samples=500, n_features=2, centers=3, random_state=42)
ds_raw = NormalizedDataset(x, y, normalize=False)
ds_norm = NormalizedDataset(x, y, normalize=True)

print(f"原始数据: 均值={ds_raw.features.mean(dim=0)}, 标准差={ds_raw.features.std(dim=0)}")
print(f"归一化后: 均值={ds_norm.features.mean(dim=0)}, 标准差={ds_norm.features.std(dim=0)}")
print()

# ============================================================
print("=" * 60)
print("6.3 训练/验证/测试集划分")
print("=" * 60)

from sklearn.model_selection import train_test_split


def train_val_test_split(features, labels, train_r=0.7, val_r=0.15, test_r=0.15):
    """划分训练/验证/测试集"""
    assert abs(train_r + val_r + test_r - 1.0) < 1e-6

    # 先分离测试集
    x_temp, x_test, y_temp, y_test = train_test_split(
        features, labels, test_size=test_r, random_state=42, stratify=labels
    )
    # 再分离训练集和验证集
    val_ratio = val_r / (train_r + val_r)
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
    )

    print(f"训练集: {len(x_train)}, 验证集: {len(x_val)}, 测试集: {len(x_test)}")
    return x_train, x_val, x_test, y_train, y_val, y_test


# 生成数据并划分
x_raw, y_raw = make_blobs(n_samples=600, n_features=2, centers=3, random_state=42)
x_tr, x_va, x_te, y_tr, y_va, y_te = train_val_test_split(x_raw, y_raw)

# 创建三个 Dataset
train_ds = NormalizedDataset(x_tr, y_tr)
val_ds = NormalizedDataset(x_va, y_va)
test_ds = NormalizedDataset(x_te, y_te)

print(f"Dataset 数量: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
print()

# ============================================================
print("=" * 60)
print("6.4 完整的分类流程（用 Dataset 重构）")
print("=" * 60)


class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_one_epoch(model, dataset, optimizer, criterion):
    """遍历整个数据集训练一次（不用 DataLoader）"""
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0

    for i in range(len(dataset)):
        x, y = dataset[i]
        x, y = x.unsqueeze(0), y.unsqueeze(0)  # 假装 batch=1

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = logits.argmax(dim=1)
        total_correct += (pred == y).sum().item()
        total_samples += 1
        total_loss += loss.item()

    return total_loss / total_samples, total_correct / total_samples


def evaluate(model, dataset, criterion):
    """评估（不需要梯度）"""
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0

    with torch.no_grad():
        for i in range(len(dataset)):
            x, y = dataset[i]
            x, y = x.unsqueeze(0), y.unsqueeze(0)

            logits = model(x)
            loss = criterion(logits, y)

            pred = logits.argmax(dim=1)
            total_correct += (pred == y).sum().item()
            total_samples += 1
            total_loss += loss.item()

    return total_loss / total_samples, total_correct / total_samples


# 初始化
model = Classifier(in_dim=2, hidden_dim=64, num_classes=3)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练循环
print("训练开始 (每个样本单独训练，效率低，下节用 DataLoader 改进):")
for epoch in range(30):
    train_loss, train_acc = train_one_epoch(model, train_ds, optimizer, criterion)
    val_loss, val_acc = evaluate(model, val_ds, criterion)

    if (epoch + 1) % 10 == 0:
        print(f"epoch {epoch+1:2d}: "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}")

test_loss, test_acc = evaluate(model, test_ds, criterion)
print(f"\n测试集: loss={test_loss:.4f}, acc={test_acc:.3f}")

# ============================================================
print("\n" + "=" * 60)
print("6.5 常见 Dataset 变体")
print("=" * 60)

print("""
  1. Map-style Dataset（本节课所讲）
     - 实现 __getitem__ 和 __len__
     - 支持随机访问 dataset[i]
     - 适合：表格数据、图像文件列表

  2. Iterable-style Dataset
     - 实现 __iter__
     - 不支持随机访问
     - 适合：流式数据、超大数据集、实时数据

  3. 内置 Dataset（下一节 DataLoader 会用到）
     - TensorDataset:  包装 tensor
     - ConcatDataset:  拼接多个 Dataset
     - Subset:         取子集
     - random_split:   随机划分
""")

print("=" * 60)
print("✅ 第 6 步完成！")
print("=" * 60)
