"""
======================================================================
第 11 步：学习率衰减（Learning Rate Scheduling）
======================================================================
学习目标：
  1. 理解学习率对训练的影响（过大/过小/适中）
  2. 掌握各种 scheduler 的用法与适用场景
  3. 理解 warmup 的原理
  4. 可视化不同 scheduler 的学习率变化曲线

关键词：StepLR, CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR, Warmup
======================================================================

为什么需要学习率衰减？
  - 训练初期：大学习率快速收敛
  - 训练后期：小学习率精细调优
  - 避免在最优解附近震荡
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import numpy as np
from utils import set_seed

set_seed(42)

# ============================================================
print("=" * 60)
print("11.1 学习率对训练的影响")
print("=" * 60)

def test_lr_values():
    """演示不同固定学习率的效果"""
    def f(x):
        return x**2  # 最小值在 x=0

    lrs = [0.01, 0.1, 0.5, 1.0, 2.0]
    fig, axes = plt.subplots(1, len(lrs), figsize=(15, 3))

    for ax, lr in zip(axes, lrs):
        x = torch.tensor(2.0, requires_grad=True)
        xs = [x.item()]

        for _ in range(30):
            y = f(x)
            y.backward()
            with torch.no_grad():
                x -= lr * x.grad
            x.grad.zero_()
            xs.append(x.item())

        ax.plot(xs, 'b-o', markersize=4)
        ax.axhline(0, color='r', linestyle='--', alpha=0.5)
        ax.set_title(f"lr={lr}")
        ax.set_xlabel("Step")
        ax.set_ylabel("x")

    plt.suptitle("不同学习率的收敛效果", fontsize=14)
    plt.tight_layout()
    plt.savefig("data/lr_comparison.png", dpi=150)
    plt.close()
    print("图片已保存到: data/lr_comparison.png")
    print("结论: lr 太小收敛慢，lr 太大可能振荡/发散")
    print()

test_lr_values()

# ============================================================
print("=" * 60)
print("11.2 常用 Scheduler 详解")
print("=" * 60)

# 创建一个简单模型来演示
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 3),
)

# 用代码可视化各种 scheduler 的学习率变化
total_epochs = 100
lrs_to_plot = {}  # 存储各组 lr 数值

# --- 1. StepLR: 每 step_size 个 epoch 乘以 gamma ---
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
lrs_step = []
for epoch in range(total_epochs):
    lrs_step.append(optimizer.param_groups[0]['lr'])
    optimizer.step()
    scheduler.step()

# --- 2. MultiStepLR: 在指定 milestone 处调整 ---
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.5)
lrs_multi = []
for epoch in range(total_epochs):
    lrs_multi.append(optimizer.param_groups[0]['lr'])
    optimizer.step()
    scheduler.step()

# --- 3. ExponentialLR: 每个 epoch 乘以 gamma ---
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
lrs_exp = []
for epoch in range(total_epochs):
    lrs_exp.append(optimizer.param_groups[0]['lr'])
    optimizer.step()
    scheduler.step()

# --- 4. CosineAnnealingLR: 余弦衰减 ---
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=0)
lrs_cos = []
for epoch in range(total_epochs):
    lrs_cos.append(optimizer.param_groups[0]['lr'])
    optimizer.step()
    scheduler.step()

# --- 5. CosineAnnealingWarmRestarts: 带热重启的余弦 ---
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
lrs_cos_warm = []
for epoch in range(total_epochs):
    lrs_cos_warm.append(optimizer.param_groups[0]['lr'])
    optimizer.step()
    scheduler.step()

# --- 6. ReduceLROnPlateau: 根据指标自动调整 ---
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, verbose=False
)
lrs_plateau = [0.1]
# 模拟验证损失下降后停滞
val_losses = np.concatenate([
    np.linspace(1.0, 0.3, 30),  # 快速下降
    np.full(40, 0.3),            # 平台期
    np.full(30, 0.3),            # 持续平台
])
for epoch in range(total_epochs):
    # ReduceLROnPlateau 需要手动调用并传入指标
    scheduler.step(val_losses[epoch])
    lrs_plateau.append(optimizer.param_groups[0]['lr'])

lrs_plateau = lrs_plateau[:total_epochs]  # 保持长度一致

# --- 7. OneCycleLR: 先升后降 ---
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.1, steps_per_epoch=1, epochs=total_epochs,
    pct_start=0.3,  # 前 30% 时间做 warmup
)
lrs_onecycle = []
for epoch in range(total_epochs):
    lrs_onecycle.append(optimizer.param_groups[0]['lr'])
    optimizer.step()
    scheduler.step()

# ============================================================
print("11.3 Scheduler 可视化对比")
print("=" * 60)

schedulers = {
    'StepLR (step=30, γ=0.1)': lrs_step,
    'MultiStepLR ([30,60,80], γ=0.5)': lrs_multi,
    'ExponentialLR (γ=0.95)': lrs_exp,
    'CosineAnnealingLR': lrs_cos,
    'CosineAnnealingWarmRestarts': lrs_cos_warm,
    'ReduceLROnPlateau': lrs_plateau,
    'OneCycleLR': lrs_onecycle,
}

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle("各种学习率调度器对比", fontsize=14)

for ax, (name, lrs) in zip(axes.flat, schedulers.items()):
    ax.plot(lrs, 'b-', linewidth=1.5)
    ax.set_title(name, fontsize=10)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('LR')
    ax.grid(True, alpha=0.3)

# 最后一个子图为总对比图
axes[1, 3].cla()
for name, lrs in schedulers.items():
    axes[1, 3].plot(lrs, linewidth=1, alpha=0.7, label=name.split('(')[0])
axes[1, 3].set_title('All Schedulers')
axes[1, 3].set_xlabel('Epoch')
axes[1, 3].set_ylabel('LR')
axes[1, 3].legend(fontsize=7)

plt.tight_layout()
plt.savefig("data/lr_schedulers_comparison.png", dpi=150)
plt.close()
print("图片已保存到: data/lr_schedulers_comparison.png")
print()

# ============================================================
print("=" * 60)
print("11.4 Warmup 预热机制")
print("=" * 60)


class WarmupScheduler:
    """自定义 Warmup: 先从小 lr 线性增长，再切换正式 scheduler"""
    def __init__(self, optimizer, warmup_epochs, after_scheduler):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.after_scheduler = after_scheduler
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            # 线性增长
            scale = self.current_epoch / self.warmup_epochs
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg['lr'] = base_lr * scale
        else:
            self.after_scheduler.step()

    def state_dict(self):
        return {
            'current_epoch': self.current_epoch,
            'after_scheduler': self.after_scheduler.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.current_epoch = state_dict['current_epoch']
        self.after_scheduler.load_state_dict(state_dict['after_scheduler'])


# 可视化 Warmup
model2 = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 3))
optimizer = optim.Adam(model2.parameters(), lr=0.001)
after_sched = lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-6)
warmup_sched = WarmupScheduler(optimizer, warmup_epochs=10, after_scheduler=after_sched)

lrs_warmup_demo = []
for _ in range(100):
    lrs_warmup_demo.append(optimizer.param_groups[0]['lr'])
    warmup_sched.step()

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(lrs_warmup_demo)
ax.axvline(10, color='r', linestyle='--', label='Warmup 结束')
ax.set_xlabel('Epoch')
ax.set_ylabel('Learning Rate')
ax.set_title('Warmup (10 epochs) + CosineAnnealingLR')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig("data/lr_warmup.png", dpi=150)
plt.close()
print("Warmup 可视化已保存到: data/lr_warmup.png")
print("Warmup 原因: 训练初期参数随机，过大的 lr 可能导致训练不稳定")
print()

# ============================================================
print("=" * 60)
print("11.5 Scheduler 选择指南")
print("=" * 60)
print("""
  ┌─────────────────────┬──────────────────────────────────────┐
  │ Scheduler           │ 适用场景                              │
  ├─────────────────────┼──────────────────────────────────────┤
  │ StepLR              │ 简单直接，需要调参（步子，衰减因子）    │
  │ MultiStepLR         │ 明确知道何时降 lr                       │
  │ ExponentialLR       │ 持续、平滑的衰减                       │
  │ CosineAnnealingLR   │ ✅ 最常用 — 平滑衰减到 0               │
  │ CosineWarmRestarts  │ 超参搜索、集成学习                      │
  │ ReduceLROnPlateau   │ 不知道何时衰减，让指标说话              │
  │ OneCycleLR          │ 快速训练、大 batch 训练                 │
  │ LambdaLR            │ 自定义任意衰减函数                      │
  │ ConstantLR          │ 前几轮用固定小 lr（warmup 特例）        │
  └─────────────────────┴──────────────────────────────────────┘

  推荐策略:
  - 入门: StepLR / CosineAnnealingLR
  - 调优: ReduceLROnPlateau
  - 竞赛: CosineWarmRestarts + Warmup
  - 发现瓶颈才换 scheduler，不要频繁换
  - 配合 Warmup（5-10 epochs）几乎总有帮助

  使用方式:
  scheduler.step()                  # 每个 epoch 后调用
  scheduler.step(val_loss)          # ReduceLROnPlateau 传入指标
""")

print("=" * 60)
print("✅ 第 11 步完成！")
print("=" * 60)
