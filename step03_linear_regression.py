"""
======================================================================
第 3 步：线性回归 — 原始实现 vs 封装实现
======================================================================
学习目标：
  1. 手动实现线性回归（纯张量 + 梯度下降）
  2. 使用 nn.Module + nn.Linear + optimizer 封装实现
  3. 对比两种方式，理解框架的封装与便利性
  4. 掌握训练循环（epoch, forward, loss, backward, update）的五步流程

关键词：nn.Linear, SGD, MSELoss, optimizer, 训练循环
======================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import set_seed

set_seed(42)


def make_linear_data(n=100, w_true=2.0, b_true=0.5, noise=0.5):
    """生成线性回归数据: y = wx + b + 噪声"""
    x = torch.linspace(-3, 3, n).reshape(-1, 1)
    y = w_true * x + b_true + torch.randn_like(x) * noise
    return x, y


print("=" * 60)
print(f"3.1 线性回归 — 原始实现（纯张量 + 手动梯度）")
print("=" * 60)

# 准备数据
x, y_true = make_linear_data(100)
print(f"数据: x.shape={x.shape}, y.shape={y_true.shape}")
print(f"真实参数: w=2.0, b=0.5")

# 初始化参数（需要梯度）
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

print(f"\n初始: w={w.item():.4f}, b={b.item():.4f}")

# 超参数
lr = 0.05
epochs = 200

# 训练历史
losses = []
w_history, b_history = [w.item()], [b.item()]

for epoch in range(epochs):
    # ----- 1. 前向传播 -----
    y_pred = w * x + b

    # ----- 2. 计算损失 (MSE) -----
    loss = ((y_pred - y_true) ** 2).mean()

    # ----- 3. 反向传播 -----
    loss.backward()

    # ----- 4. 更新参数（梯度下降）-----
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    # ----- 5. 清零梯度 -----
    w.grad.zero_()
    b.grad.zero_()

    losses.append(loss.item())
    w_history.append(w.item())
    b_history.append(b.item())

    if (epoch + 1) % 50 == 0:
        print(f"epoch {epoch+1:3d}: loss={loss.item():.4f}, "
              f"w={w.item():.3f}, b={b.item():.3f}")

print(f"\n最终: w={w.item():.3f}, b={b.item():.3f}")
print(f"真实: w=2.000, b=0.500")

print("\n原始实现的要点:")
print("  - 手动管理参数的 requires_grad")
print("  - 手动编写梯度下降更新代码")
print("  - 手动清零梯度（容易忘记）")
print("  - 参数多时，代码会非常冗长")


print("\n" + "=" * 60)
print(f"3.2 线性回归 — 封装实现（nn.Module + optimizer）")
print("=" * 60)


class LinearRegression(nn.Module):
    """用 nn.Module 封装线性回归模型"""
    def __init__(self, in_features=1, out_features=1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


# 准备数据
x, y_true = make_linear_data(100)

# 模型、损失函数、优化器
model = LinearRegression()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.05)

print(f"模型结构:\n{model}")
print(f"初始: weight={model.linear.weight.item():.4f}, bias={model.linear.bias.item():.4f}")

# 训练
epochs = 200
losses2 = []
w_hist2, b_hist2 = [model.linear.weight.item()], [model.linear.bias.item()]

for epoch in range(epochs):
    # ----- 1. 前向传播 -----
    y_pred = model(x)

    # ----- 2. 计算损失 -----
    loss = criterion(y_pred, y_true)

    # ----- 3. 反向传播 -----
    optimizer.zero_grad()  # 清零梯度
    loss.backward()

    # ----- 4. 更新参数 -----
    optimizer.step()

    losses2.append(loss.item())
    w_hist2.append(model.linear.weight.item())
    b_hist2.append(model.linear.bias.item())

    if (epoch + 1) % 50 == 0:
        print(f"epoch {epoch+1:3d}: loss={loss.item():.4f}, "
              f"w={model.linear.weight.item():.3f}, "
              f"b={model.linear.bias.item():.3f}")

print(f"\n最终: w={model.linear.weight.item():.3f}, b={model.linear.bias.item():.3f}")
print(f"真实: w=2.000, b=0.500")


print("\n" + "=" * 60)
print(f"3.3 对比可视化")
print("=" * 60)

# 可视化训练结果
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("线性回归 — 原始 vs 封装对比", fontsize=14)

# 拟合直线
axes[0].scatter(x.numpy(), y_true.numpy(), alpha=0.5, s=15, label="数据点")
axes[0].plot(x.numpy(), (w.item() * x + b.item()).numpy(),
             'r-', linewidth=2, label=f"原始: y={w.item():.2f}x+{b.item():.2f}")
axes[0].plot(x.numpy(), model(x).detach().numpy(),
             'g--', linewidth=2, label=f"封装: y={model.linear.weight.item():.2f}x+{model.linear.bias.item():.2f}")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].set_title("数据与拟合直线")
axes[0].legend()

# 损失曲线
axes[1].plot(losses, alpha=0.7, label="原始实现")
axes[1].plot(losses2, alpha=0.7, label="封装实现")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss (MSE)")
axes[1].set_title("损失下降曲线")
axes[1].legend()

# 参数轨迹
axes[2].plot(w_hist2, b_hist2, 'b-', alpha=0.6, label="封装参数轨迹")
axes[2].plot(w_history, b_history, 'r--', alpha=0.6, label="原始参数轨迹")
axes[2].scatter([2.0], [0.5], c='green', s=100, marker='*', label="真实参数", zorder=5)
axes[2].set_xlabel("w")
axes[2].set_ylabel("b")
axes[2].set_title("参数轨迹")
axes[2].legend()

plt.tight_layout()
plt.savefig("data/linear_regression_comparison.png", dpi=150)
plt.close()
print("图片已保存到: data/linear_regression_comparison.png")

print("\n" + "=" * 60)
print("3.4 训练循环五步法总结")
print("=" * 60)
print("""
  1. y_pred = model(x)           # 前向传播
  2. loss = criterion(y_pred, y) # 计算损失
  3. optimizer.zero_grad()       # 清零梯度 ⚠️ 容易忘
  4. loss.backward()             # 反向传播
  5. optimizer.step()            # 更新参数

  numpy 对照版:
  1. y_pred = w @ x + b
  2. loss = np.mean((y_pred - y)²)
  3. （不需要，numpy 不累积梯度）
  4. dw = np.mean(2*(y_pred-y)*x, axis=1)  # 手动求导
     db = np.mean(2*(y_pred-y))
  5. w -= lr * dw
     b -= lr * db
""")

print("=" * 60)
print("✅ 第 3 步完成！")
print("=" * 60)
