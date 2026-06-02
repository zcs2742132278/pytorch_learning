"""
======================================================================
第 7 步：DataLoader 重构
======================================================================
学习目标：
  1. 掌握 DataLoader 的核心参数（batch_size, shuffle, num_workers）
  2. 理解 collate_fn 的作用
  3. 将第 6 步的逐样本训练改为批量训练
  4. 体验 batch 训练 vs 逐样本训练的性能差异
  5. 理解 num_workers 多进程数据加载

关键词：DataLoader, batch_size, shuffle, collate_fn, num_workers
======================================================================

DataLoader 是 PyTorch 的数据加载引擎：
  Dataset   → 定义"单个样本怎么取"
  DataLoader → 定义"一批样本怎么组织"
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from sklearn.datasets import make_blobs
import time
from utils import set_seed

set_seed(42)

# ============================================================
print("=" * 60)
print("7.1 DataLoader 基础用法")
print("=" * 60)

# 准备数据
x_raw, y_raw = make_blobs(n_samples=1000, n_features=10, centers=3, random_state=42)
x_t = torch.FloatTensor(x_raw)
y_t = torch.LongTensor(y_raw)
dataset = TensorDataset(x_t, y_t)  # 便捷包装

print(f"数据集大小: {len(dataset)}")

# 创建 DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 获取一个 batch
batch_x, batch_y = next(iter(loader))
print(f"\n一个 batch:")
print(f"  batch_x: shape={batch_x.shape}, dtype={batch_x.dtype}")
print(f"  batch_y: shape={batch_y.shape}, dtype={batch_y.dtype}")
print(f"  每个 epoch 有 {len(loader)} 个 batch")

# 遍历所有 batch
print(f"\n各 batch 的 x 形状:")
for i, (xb, yb) in enumerate(loader):
    if i < 3:
        print(f"  batch {i}: x.shape={xb.shape}, y 类别分布={yb.bincount()}")
    # 最后一个 batch 可能不满
    if i == len(loader) - 1:
        print(f"  batch {i} (最后一个): x.shape={xb.shape}")
print()

# ============================================================
print("=" * 60)
print("7.2 核心参数详解")
print("=" * 60)

# --- batch_size ---
print("batch_size 的影响:")
for bs in [1, 16, 64, 256]:
    ld = DataLoader(dataset, batch_size=bs)
    print(f"  batch_size={bs:3d}: {len(ld)} 个 batch")

print("\n--- shuffle ---")
print("  shuffle=True  : 每个 epoch 打乱数据顺序（训练时必须）")
print("  shuffle=False : 保持原始顺序（验证/测试时推荐）")
print("  ⚠️ 没有 shuffle，模型可能学到数据顺序的偏差")

print("\n--- drop_last ---")
ld_drop = DataLoader(dataset, batch_size=64, drop_last=True)
ld_keep = DataLoader(dataset, batch_size=64, drop_last=False)
print(f"  drop_last=True  : {len(ld_drop)} 个 batch（丢弃最后不完整的 batch）")
print(f"  drop_last=False : {len(ld_keep)} 个 batch（保留最后不完整的 batch）")

print("\n--- num_workers ---")
print("  num_workers=0 (默认): 主进程加载数据")
print("  num_workers>0:         多子进程并行加载（加速大 I/O）")
print("  ⚠️ Windows 下需用 if __name__ == '__main__' 保护")
print("  推荐值: 4-8（取决于 CPU 核心数和 I/O 瓶颈）")

print("\n--- pin_memory ---")
print("  pin_memory=True: 加速 CPU→GPU 数据传输（需配合 non_blocking）")
print()

# ============================================================
print("=" * 60)
print("7.3 批量训练 vs 逐样本训练（性能对比）")
print("=" * 60)


class MLP(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# 准备数据
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

# --- 方式1: 批量训练 ---
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

model_batch = MLP(10, 3)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_batch.parameters(), lr=0.01)

print("批量训练 (batch_size=64):")
t0 = time.time()
for epoch in range(50):
    model_batch.train()
    for xb, yb in train_loader:
        logits = model_batch(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
time_batch = time.time() - t0

# 评估
model_batch.eval()
correct, total = 0, 0
with torch.no_grad():
    for xb, yb in val_loader:
        pred = model_batch(xb).argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += len(yb)
print(f"  耗时: {time_batch:.3f}s, 准确率: {correct/total:.3f}")

# --- 方式2: 逐样本训练（对比）---
model_ones = MLP(10, 3)
optimizer = torch.optim.Adam(model_ones.parameters(), lr=0.01)

print("\n逐样本训练 (每次一个样本):")
t0 = time.time()
for epoch in range(50):
    model_ones.train()
    # 手动遍历（不用 DataLoader）
    train_indices = torch.randperm(len(train_ds))  # 手动打乱
    for idx in train_indices:
        x, y = train_ds[idx]
        logits = model_ones(x.unsqueeze(0))
        loss = criterion(logits, y.unsqueeze(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
time_ones = time.time() - t0
print(f"  耗时: {time_ones:.3f}s")

print(f"\n⚡ 批量训练加速: {time_ones / time_batch:.1f}×")
print("  （实际项目中差距更大，因为 GPU 并行 + 内存局部性）")
print()

# ============================================================
print("=" * 60)
print("7.4 collate_fn — 自定义 batch 组织逻辑")
print("=" * 60)


class VariableLengthDataset(Dataset):
    """变长序列数据集"""
    def __init__(self, n_samples=100):
        self.data = [
            torch.randn(np.random.randint(3, 10))  # 长度 3~9
            for _ in range(n_samples)
        ]
        self.labels = torch.randint(0, 3, (n_samples,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def pad_collate_fn(batch):
    """自定义 collate: 将变长序列填充到相同长度"""
    sequences, labels = zip(*batch)

    # 计算最大长度
    max_len = max(len(seq) for seq in sequences)

    # 填充
    padded = torch.zeros(len(sequences), max_len)
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = seq

    labels = torch.tensor(labels)
    return padded, labels


var_ds = VariableLengthDataset(20)
var_loader = DataLoader(var_ds, batch_size=5, collate_fn=pad_collate_fn)

xb, yb = next(iter(var_loader))
print(f"变长序列填充后: x.shape={xb.shape} (均已填充到最长)")
print(f"  labels: {yb}")
print()

# PyTorch 内置的 collate 函数
from torch.utils.data import default_collate
print(f"默认的 default_collate 会自动将 List[Tensor] 堆叠成 Tensor")

# ============================================================
print("=" * 60)
print("7.5 完整的训练 + 验证 + 测试流程")
print("=" * 60)

# 三层 Dataset
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
# 测试集（模拟）
test_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

model = MLP(10, 3)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print(f"{'Epoch':<6} {'Train Loss':<12} {'Train Acc':<10} {'Val Loss':<12} {'Val Acc':<10}")
print("-" * 50)

for epoch in range(30):
    # ---- 训练 ----
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    for xb, yb in train_loader:
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * len(yb)
        train_correct += (logits.argmax(1) == yb).sum().item()
        train_total += len(yb)

    # ---- 验证 ----
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb)
            loss = criterion(logits, yb)
            val_loss += loss.item() * len(yb)
            val_correct += (logits.argmax(1) == yb).sum().item()
            val_total += len(yb)

    if (epoch + 1) % 10 == 0:
        print(f"{epoch+1:<6} "
              f"{train_loss/train_total:<12.4f} "
              f"{train_correct/train_total:<10.3f} "
              f"{val_loss/val_total:<12.4f} "
              f"{val_correct/val_total:<10.3f}")

print("\n" + "=" * 60)
print("7.6 DataLoader 完整参数清单")
print("=" * 60)
print("""
  DataLoader(dataset, batch_size=1, shuffle=False,
             sampler=None, batch_sampler=None,
             num_workers=0, collate_fn=None,
             pin_memory=False, drop_last=False,
             timeout=0, worker_init_fn=None,
             prefetch_factor=2, persistent_workers=False)

  参数速查:
  - batch_size:     每个 batch 的样本数
  - shuffle:         是否打乱（训练=True, 验证=False）
  - num_workers:     子进程数（0=主进程, >0=多进程）
  - collate_fn:      如何将样本列表合并为 batch（默认堆叠）
  - pin_memory:     锁页内存，加速 CPU→GPU
  - drop_last:      丢弃最后不完整的 batch
  - prefetch_factor: 每个 worker 预取的 batch 数
  - persistent_workers: worker 进程复用（避免重复创建）
  - sampler:        自定义采样策略（用于分布式训练等）
""")

print("=" * 60)
print("✅ 第 7 步完成！")
print("=" * 60)
