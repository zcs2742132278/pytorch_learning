"""
======================================================================
第 4 步：实现分类（逻辑回归 + Softmax）
======================================================================
学习目标：
  1. 理解分类与回归的本质区别
  2. 掌握二分类（Sigmoid + BCELoss）的实现
  3. 掌握多分类（Softmax + CrossEntropyLoss）的实现
  4. 理解 Logits → Softmax → CrossEntropy 的关系
  5. 学会计算分类准确率

关键词：Sigmoid, Softmax, BCELoss, CrossEntropyLoss, accuracy
======================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from utils import set_seed

set_seed(42)

# ============================================================
# 数据准备
# ============================================================
print("=" * 60)
print("4.1 生成分类数据集")
print("=" * 60)

# 生成二分类数据
x_binary, y_binary = make_classification(
    n_samples=500, n_features=2, n_redundant=0,
    n_clusters_per_class=1, random_state=42
)

# 生成多分类数据
x_multi, y_multi = make_blobs(
    n_samples=600, n_features=2, centers=3,
    cluster_std=1.5, random_state=42
)

# 转为张量
x_binary_t = torch.FloatTensor(x_binary)
y_binary_t = torch.FloatTensor(y_binary).reshape(-1, 1)

x_multi_t = torch.FloatTensor(x_multi)
y_multi_t = torch.LongTensor(y_multi)

print(f"二分类: X{x_binary_t.shape}, y{y_binary_t.shape}, 类别数=2")
print(f"多分类: X{x_multi_t.shape}, y{y_multi_t.shape}, 类别数={len(torch.unique(y_multi_t))}")


# ============================================================
# 二分类
# ============================================================
print("\n" + "=" * 60)
print("4.2 二分类 — Sigmoid + BCELoss")
print("=" * 60)


class BinaryClassifier(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)  # 输出 1 个 logit

    def forward(self, x):
        return self.linear(x)  # 返回 logits，不包含 sigmoid


model = BinaryClassifier(in_features=2)
criterion = nn.BCEWithLogitsLoss()  # 内部包含 Sigmoid，比 BCELoss 更数值稳定
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 记录
losses, accs = [], []

for epoch in range(500):
    # 前向传播
    logits = model(x_binary_t)           # 原始输出
    loss = criterion(logits, y_binary_t) # BCEWithLogitsLoss = Sigmoid + BCE

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 预测与准确率
    with torch.no_grad():
        prob = torch.sigmoid(logits)       # 转为概率
        y_pred = (prob > 0.5).float()      # 阈值 0.5 分类
        acc = (y_pred == y_binary_t).float().mean()

    losses.append(loss.item())
    accs.append(acc.item())

    if (epoch + 1) % 100 == 0:
        print(f"epoch {epoch+1:3d}: loss={loss.item():.4f}, acc={acc.item():.3f}")

print(f"\n最终准确率: {accs[-1]:.3f}")


# ============================================================
# 多分类
# ============================================================
print("\n" + "=" * 60)
print("4.3 多分类 — Softmax + CrossEntropyLoss")
print("=" * 60)


class MultiClassifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)  # 输出 num_classes 个 logits

    def forward(self, x):
        return self.linear(x)  # 返回 logits，不包含 softmax


model_multi = MultiClassifier(in_features=2, num_classes=3)
criterion_multi = nn.CrossEntropyLoss()  # 内部 = Softmax + NLLLoss
optimizer_multi = optim.SGD(model_multi.parameters(), lr=0.1)

losses_m, accs_m = [], []

for epoch in range(500):
    logits = model_multi(x_multi_t)
    loss = criterion_multi(logits, y_multi_t)  # 交叉熵

    optimizer_multi.zero_grad()
    loss.backward()
    optimizer_multi.step()

    with torch.no_grad():
        y_pred = logits.argmax(dim=1)           # 取最大 logit 的索引
        acc = (y_pred == y_multi_t).float().mean()

    losses_m.append(loss.item())
    accs_m.append(acc.item())

    if (epoch + 1) % 100 == 0:
        print(f"epoch {epoch+1:3d}: loss={loss.item():.4f}, acc={acc.item():.3f}")

print(f"\n最终准确率: {accs_m[-1]:.3f}")

# ============================================================
# 深入理解
# ============================================================
print("\n" + "=" * 60)
print("4.4 深入理解 Softmax + CrossEntropy")
print("=" * 60)

# 手动实现 Softmax
logits = torch.tensor([[2.0, 1.0, 0.1]])
print(f"Logits: {logits}")

softmax = torch.exp(logits) / torch.exp(logits).sum(dim=1, keepdim=True)
print(f"Softmax 手动: {softmax}")
print(f"Softmax 内置: {torch.softmax(logits, dim=1)}")
print(f"概率和为 1: {softmax.sum()}")

# 手动实现 CrossEntropy
y_true = torch.tensor([0])  # 真实标签
ce_manual = -torch.log(softmax[0, y_true])
ce_builtin = nn.functional.cross_entropy(logits, y_true)
print(f"\nCrossEntropy 手动: {ce_manual.item():.4f}")
print(f"CrossEntropy 内置: {ce_builtin.item():.4f}")

print("\n⚠️ 关键理解:")
print("  nn.CrossEntropyLoss() 输入的是 logits（未经过 softmax）")
print("  损失函数内部会先做 softmax，再计算交叉熵")
print("  不要先做 softmax 再传入 CrossEntropyLoss，会出错！")


# ============================================================
# 可视化
# ============================================================
print("\n" + "=" * 60)
print("4.5 分类决策边界可视化")
print("=" * 60)


def plot_decision_boundary(model, x, y, title, ax, is_binary=True):
    """绘制二分类决策边界"""
    x_np = x.numpy()
    y_np = y.numpy().flatten() if y.dim() == 2 else y.numpy()

    # 创建网格
    x_min, x_max = x_np[:, 0].min() - 1, x_np[:, 0].max() + 1
    y_min, y_max = x_np[:, 1].min() - 1, x_np[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    with torch.no_grad():
        if is_binary:
            zz = torch.sigmoid(model(grid)).numpy()
            zz = (zz > 0.5).astype(int).reshape(xx.shape)
        else:
            zz = model(grid).argmax(dim=1).numpy().reshape(xx.shape)

    ax.contourf(xx, yy, zz, alpha=0.3, cmap='RdYlBu')
    ax.scatter(x_np[:, 0], x_np[:, 1], c=y_np, cmap='RdYlBu',
               edgecolors='k', s=20, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

plot_decision_boundary(model, x_binary_t, y_binary_t,
                       "二分类 (Sigmoid + BCEWithLogitsLoss)", axes[0], is_binary=True)
plot_decision_boundary(model_multi, x_multi_t, y_multi_t,
                       "三分类 (Softmax + CrossEntropyLoss)", axes[1], is_binary=False)

plt.tight_layout()
plt.savefig("data/classification_decision_boundary.png", dpi=150)
plt.close()
print("图片已保存到: data/classification_decision_boundary.png")

print("\n" + "=" * 60)
print("4.6 损失函数速查表")
print("=" * 60)
print("""
  二分类（标签: 0 或 1）:
    nn.BCEWithLogitsLoss()   ← ✅ 推荐（数值稳定）
    nn.BCELoss()              ← 需要先手动做 Sigmoid

  多分类（标签: 整数索引）:
    nn.CrossEntropyLoss()    ← ✅ 推荐（输入 logits）
    nn.NLLLoss()              ← 需要先手动做 LogSoftmax

  多标签分类（标签: one-hot/multi-hot 向量）:
    nn.BCEWithLogitsLoss()   ← 每个类别独立做二分类

  回归:
    nn.MSELoss()              ← 均方误差
    nn.L1Loss()               ← 平均绝对误差
    nn.SmoothL1Loss()         ← Huber Loss（鲁棒）
""")

print("=" * 60)
print("✅ 第 4 步完成！")
print("=" * 60)
