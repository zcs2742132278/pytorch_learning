"""
======================================================================
第 9 步：代码校验 + 封装 — 构建规范的训练框架
======================================================================
学习目标：
  1. 掌握梯度检查、过拟合小批量测试等代码校验方法
  2. 构建可复用的 Trainer 类封装训练逻辑
  3. 添加早停（Early Stopping）机制
  4. 建立规范的日志记录和指标追踪

关键词：梯度检查, 过拟合测试, Trainer 封装, Early Stopping, 指标追踪
======================================================================

在开始复杂的 CNN 项目之前，先学会验证代码的正确性。
常见 Bug：
  - 梯度没清零
  - 反向传播之前做了 in-place 操作
  - 训练/评估模式没切换
  - 损失函数输入了错误的 shape
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from utils import set_seed

set_seed(42)

# ============================================================
print("=" * 60)
print("9.1 代码校验方法 1: 过拟合小批量测试")
print("=" * 60)
print("""
原理: 模型应该能在少量样本上把损失降到接近 0
如果不能 → 模型容量不够或代码有 Bug
如果能   → 至少前向/反向/参数更新流程正确
""")


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        return self.net(x)


# 只取 5 个样本
x = torch.randn(5, 10)
y = torch.randint(0, 3, (5,))
dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=5)

model = SimpleMLP()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("过拟合小批量测试 (5 个样本):")
for epoch in range(200):
    for xb, yb in loader:
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 50 == 0:
        acc = (logits.argmax(1) == yb).float().mean()
        print(f"  epoch {epoch+1:3d}: loss={loss.item():.6f}, acc={acc:.3f}")

if loss.item() < 0.01:
    print("✅ 过拟合小批量测试通过！代码流程正确")
else:
    print("❌ 过拟合小批量测试失败！检查代码")
print()

# ============================================================
print("=" * 60)
print("9.2 代码校验方法 2: 梯度检查")
print("=" * 60)


def check_gradient_flow(model):
    """检查模型中各参数的梯度是否正常"""
    print("梯度流动检查:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            status = "✅" if grad_norm > 1e-8 else "❌ 梯度为 0"
            if grad_norm > 100:
                status = "⚠️ 梯度爆炸"
            print(f"  {name:25s} grad_norm={grad_norm:.6f} {status}")
        else:
            print(f"  {name:25s} grad=None ❌ 无梯度")
    print()


# 做一次前向+反向
xb, yb = x, y
logits = model(xb)
loss = criterion(logits, yb)
optimizer.zero_grad()
loss.backward()

check_gradient_flow(model)

# ============================================================
print("=" * 60)
print("9.3 代码校验方法 3: 参数更新检查")
print("=" * 60)

# 记录更新前的参数
params_before = {name: param.clone()
                 for name, param in model.named_parameters()}

# 做一次更新
optimizer.step()

# 检查参数是否真的变了
print("参数更新检查:")
for name, param in model.named_parameters():
    diff = (param - params_before[name]).abs().max().item()
    status = "✅ 已更新" if diff > 1e-8 else "❌ 未变化"
    print(f"  {name:25s} max_diff={diff:.8f} {status}")
print()

# ============================================================
print("=" * 60)
print("9.4 封装: 构建可复用的 Trainer 类")
print("=" * 60)


class Trainer:
    """通用训练器 — 封装训练/验证/测试的完整流程"""

    def __init__(self, model, criterion, optimizer, device='cpu'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.model.to(self.device)

        # 训练历史
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
        }
        self.best_model_state = None
        self.best_val_acc = 0.0

    def train_epoch(self, loader):
        """训练一个 epoch"""
        self.model.train()
        total_loss, correct, total = 0, 0, 0

        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)

            # 前向
            logits = self.model(xb)
            loss = self.criterion(logits, yb)

            # 反向
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 统计
            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)

        return total_loss / total, correct / total

    @torch.no_grad()
    def evaluate(self, loader):
        """评估（不计算梯度）"""
        self.model.eval()
        total_loss, correct, total = 0, 0, 0

        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            logits = self.model(xb)
            loss = self.criterion(logits, yb)

            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)

        return total_loss / total, correct / total

    def fit(self, train_loader, val_loader, epochs,
            early_stopping_patience=None, verbose=True):
        """完整训练循环"""
        patience_counter = 0
        self.best_val_acc = 0.0

        for epoch in range(epochs):
            # 训练 + 验证
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = deepcopy(self.model.state_dict())
                patience_counter = 0
            elif early_stopping_patience:
                patience_counter += 1

            # 打印
            if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
                print(f"epoch {epoch+1:3d}/{epochs}: "
                      f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f} | "
                      f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}")

            # 早停
            if early_stopping_patience and patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"⚠️ 早停于 epoch {epoch+1}（{early_stopping_patience} 轮无提升）")
                break

        # 恢复最佳模型
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            if verbose:
                print(f"🏆 最佳验证准确率: {self.best_val_acc:.4f}")

    def plot_history(self, save_path=None):
        """绘制训练历史曲线"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # 损失
        axes[0].plot(self.history['train_loss'], label='Train')
        axes[0].plot(self.history['val_loss'], label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss Curves')
        axes[0].legend()
        axes[0].grid(True)

        # 准确率
        axes[1].plot(self.history['train_acc'], label='Train')
        axes[1].plot(self.history['val_acc'], label='Val')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy Curves')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"训练曲线已保存到: {save_path}")
        plt.close()


# ============================================================
print("=" * 60)
print("9.5 使用 Trainer 进行完整训练")
print("=" * 60)

# 生成数据
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

x_raw, y_raw = make_blobs(n_samples=1000, n_features=10, centers=4, random_state=42)
x_tr, x_va, y_tr, y_va = train_test_split(x_raw, y_raw, test_size=0.3, random_state=42)

train_ds = TensorDataset(torch.FloatTensor(x_tr), torch.LongTensor(y_tr))
val_ds = TensorDataset(torch.FloatTensor(x_va), torch.LongTensor(y_va))
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

# 模型
model = nn.Sequential(
    nn.Linear(10, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 4),
)

# 训练
trainer = Trainer(
    model=model,
    criterion=nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4),
)
trainer.fit(train_loader, val_loader, epochs=100, early_stopping_patience=15)
trainer.plot_history("data/training_history.png")

print(f"\n训练完成。训练了 {len(trainer.history['train_loss'])} 个 epoch")
print(f"最佳验证准确率: {trainer.best_val_acc:.4f}")

# ============================================================
print("\n" + "=" * 60)
print("9.6 Early Stopping 原理")
print("=" * 60)
print("""
  早停（Early Stopping）逻辑:
  1. 每个 epoch 后比较验证集指标
  2. 如果变好 → 保存当前模型，计数器归零
  3. 如果变差 → 计数器 +1
  4. 计数器达到 patience → 停止训练，恢复最佳模型

  为什么有效？
  - 防止过拟合：训练太久，验证误差会回升
  - 节省时间：不必手动猜测最佳 epoch
  - 自动确定训练终点

  典型 patience 取值:
  - 小数据集: 10-20
  - 大数据集: 5-10
  - 稳定训练: 可以设大些
""")

# ============================================================
print("=" * 60)
print("9.7 完整代码校验清单")
print("=" * 60)
print("""
  训练前检查:
  □ 过拟合小批量测试 → 模型能拟合少量数据
  □ 梯度流动检查    → 所有参数梯度非零
  □ 参数更新检查    → optimizer.step() 后参数确实变了
  □ 数据 shape 检查 → 输入输出维度匹配
  □ 模式检查        → train() vs eval() 正确切换

  训练中检查:
  □ 损失是否下降    → 持续下降说明学习正常
  □ 准确率是否提升  → 应稳步上升
  □ 训练/验证差距   → 差距过大说明过拟合
  □ 梯度范数        → 避免梯度爆炸/消失

  训练后检查:
  □ 验证集/测试集指标 → 最终评估
  □ 混淆矩阵          → 查看各类别表现
  □ 错误样本分析      → 了解模型弱点
""")

print("=" * 60)
print("✅ 第 9 步完成！")
print("=" * 60)
