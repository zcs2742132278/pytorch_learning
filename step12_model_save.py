"""
======================================================================
第 12 步：模型参数保存与加载
======================================================================
学习目标：
  1. 掌握 state_dict 的概念
  2. 掌握模型保存/加载的两种方式
  3. 实现 Checkpoint 断点续训
  4. 理解 optimizer/scheduler 状态的保存
  5. 加载预训练权重（部分加载、冻结参数）

关键词：state_dict, torch.save, torch.load, checkpoint, 断点续训
======================================================================

PyTorch 保存模型的核心概念:
  model.state_dict() → OrderedDict {参数名: 参数张量}
  torch.save(state_dict, path) → 保存到磁盘
  model.load_state_dict(torch.load(path)) → 恢复参数
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import json
from pathlib import Path
from utils import set_seed

set_seed(42)

# ============================================================
print("=" * 60)
print("12.1 state_dict 是什么？")
print("=" * 60)

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 3),
)

print("model.state_dict() 的键（部分）:")
for key in list(model.state_dict().keys()):
    tensor = model.state_dict()[key]
    print(f"  {key:20s}  shape={str(tensor.shape):12s}  dtype={tensor.dtype}")

print(f"\nstate_dict 是一个 OrderedDict，共 {len(model.state_dict())} 个条目")
print()

# optimizer 也有 state_dict
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 做一个虚拟的 step，让 optimizer 产生 state（momentum 等）
dummy_x = torch.randn(4, 10)
dummy_y = torch.randint(0, 3, (4,))
loss = nn.functional.cross_entropy(model(dummy_x), dummy_y)
loss.backward()
optimizer.step()

print("optimizer.state_dict() 的键:")
for key in optimizer.state_dict():
    print(f"  {key}")
print()

# ============================================================
print("=" * 60)
print("12.2 模型保存的两种方式")
print("=" * 60)

# 创建临时目录
save_dir = Path("data/checkpoints")
save_dir.mkdir(parents=True, exist_ok=True)

# --- 方式 1: 只保存参数（推荐）---
model_path = save_dir / "model_weights.pth"
torch.save(model.state_dict(), model_path)
print(f"✅ 方式1 (只保存参数): {model_path}")
print(f"   文件大小: {os.path.getsize(model_path):,} bytes")

# 加载方式1
loaded_model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 3),
)
loaded_model.load_state_dict(torch.load(model_path))
print("   加载成功！参数已恢复")

# --- 方式 2: 保存整个模型（不推荐）---
full_model_path = save_dir / "full_model.pth"
torch.save(model, full_model_path)
print(f"\n⚠️ 方式2 (保存整个模型): {full_model_path}")
print(f"   文件大小: {os.path.getsize(full_model_path):,} bytes")

# 加载方式2
loaded_full = torch.load(full_model_path)
print("   加载成功！但这种方式有以下问题：")
print("     - 依赖原始代码的文件结构（类定义位置不能变）")
print("     - 跨环境迁移可能失败（Python/库版本不同）")
print("     - 可读性差")

print("\n推荐：始终使用方式1（只保存 state_dict）")
print()

# ============================================================
print("=" * 60)
print("12.3 Checkpoint 断点续训")
print("=" * 60)

# 模拟训练过程（数据保持固定以便验证恢复）
x = torch.randn(200, 10)
y = torch.randint(0, 3, (200,))
train_ds = TensorDataset(x[:150], y[:150])
val_ds = TensorDataset(x[150:], y[150:])
train_loader = DataLoader(train_ds, batch_size=32)
val_loader = DataLoader(val_ds, batch_size=32)


def save_checkpoint(model, optimizer, scheduler, epoch, best_acc, path):
    """保存完整 checkpoint（模型 + 优化器 + 调度器 + 训练状态）"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_acc': best_acc,
    }
    torch.save(checkpoint, path)
    return path


def load_checkpoint(model, optimizer, scheduler, path):
    """加载 checkpoint 并恢复训练状态"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint.get('epoch', 0), checkpoint.get('best_acc', 0)


# --- 第一阶段：训练 20 个 epoch ---
model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 3))
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

print("第一阶段: 训练 20 epochs")
best_acc = 0.0
for epoch in range(20):
    model.train()
    for xb, yb in train_loader:
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 评估
    model.eval()
    with torch.no_grad():
        correct = sum((model(xb).argmax(1) == yb).sum().item()
                      for xb, yb in val_loader)
        acc = correct / len(val_ds)
        if acc > best_acc:
            best_acc = acc

print(f"  第一阶段完成, best_acc = {best_acc:.4f}")

# 保存 checkpoint
ckpt_path = save_dir / "checkpoint_epoch20.pth"
save_checkpoint(model, optimizer, None, 20, best_acc, ckpt_path)
print(f"  Checkpoint 已保存: {ckpt_path}")

# --- "意外中断" ---
del model, optimizer

# --- 第二阶段：从 checkpoint 恢复，继续训练 10 个 epoch ---
print("\n第二阶段: 从 checkpoint 恢复并继续训练 10 epochs")
model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 3))
optimizer = optim.Adam(model.parameters(), lr=0.01)

start_epoch, best_acc = load_checkpoint(model, optimizer, None, ckpt_path)
print(f"  恢复状态: epoch={start_epoch}, best_acc={best_acc:.4f}")

# 继续训练
for epoch in range(start_epoch, start_epoch + 10):
    model.train()
    for xb, yb in train_loader:
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        correct = sum((model(xb).argmax(1) == yb).sum().item()
                      for xb, yb in val_loader)
        acc = correct / len(val_ds)
        if acc > best_acc:
            best_acc = acc

print(f"  恢复后训练完成, best_acc = {best_acc:.4f}")

print("\n✅ 断点续训验证成功！从 epoch 20 恢复没有丢失进度")
print()

# ============================================================
print("=" * 60)
print("12.4 加载预训练权重（迁移学习预备）")
print("=" * 60)


# 场景：新模型比原模型多一个输出类别
class ExtendedModel(nn.Module):
    def __init__(self, old_model, new_num_classes):
        super().__init__()
        # 复用旧模型的特征提取层（除了最后一层）
        self.features = nn.Sequential(*list(old_model.children())[:-1])
        self.classifier = nn.Linear(64, new_num_classes)  # 新的分类头

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# 加载预训练权重（部分加载）
old_model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 3))
# 假设 old_model 已经在大数据集上训练过（这里跳过实际训练）

new_model = ExtendedModel(old_model, new_num_classes=5)

# 只加载匹配的层
pretrained_dict = old_model.state_dict()
model_dict = new_model.state_dict()

# 过滤掉不匹配的键
pretrained_dict = {k: v for k, v in pretrained_dict.items()
                   if k in model_dict and v.shape == model_dict[k].shape}

print("加载预训练权重（部分加载）:")
print(f"  预训练参数: {len(old_model.state_dict())} 个")
print(f"  成功加载:   {len(pretrained_dict)} 个")
print(f"  被忽略:     {len(old_model.state_dict()) - len(pretrained_dict)} 个")

model_dict.update(pretrained_dict)
new_model.load_state_dict(model_dict)
print("✅ 部分加载完成")
print()

# ============================================================
print("=" * 60)
print("12.5 最佳实践: 保存路径管理")
print("=" * 60)


class CheckpointManager:
    """Checkpoint 管理器：保存最佳 k 个 + 定期保存"""

    def __init__(self, save_dir, max_keep=5, save_best_only=False):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_keep = max_keep
        self.save_best_only = save_best_only
        self.best_acc = 0.0
        self.history = []

    def save(self, model, optimizer, scheduler, epoch, acc, loss, is_best=False):
        """保存 checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'acc': acc,
            'loss': loss,
        }

        # 定期保存
        if not self.save_best_only:
            path = self.save_dir / f"checkpoint_epoch{epoch:04d}.pth"
            torch.save(checkpoint, path)
            self.history.append(path)

        # 最佳模型额外保存
        if is_best or (acc > self.best_acc):
            self.best_acc = acc
            best_path = self.save_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"  🏆 最佳模型已保存 (acc={acc:.4f})")

        # 清理旧 checkpoint
        if len(self.history) > self.max_keep:
            old_path = self.history.pop(0)
            if old_path.exists():
                old_path.unlink()

    def load_best(self, model, optimizer=None, scheduler=None):
        best_path = self.save_dir / "best_model.pth"
        if best_path.exists():
            return load_checkpoint(model, optimizer, scheduler, best_path)
        return 0, 0.0


# 使用示例
ckpt_mgr = CheckpointManager(save_dir, max_keep=3, save_best_only=False)
print(f"CheckpointManager 已创建，保存目录: {save_dir}")
print(f"  规则：最多保留 3 个普通 checkpoint + 1 个 best_model")
print()

# ============================================================
print("=" * 60)
print("12.6 保存/加载速查表")
print("=" * 60)
print("""
  保存模型参数:
    torch.save(model.state_dict(), 'model.pth')

  加载模型参数:
    model = YourModel()
    model.load_state_dict(torch.load('model.pth'))

  保存 Checkpoint:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, 'checkpoint.pth')

  加载 Checkpoint:
    checkpoint = torch.load('checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

  CPU/GPU 兼容加载:
    # GPU 训练的模型在 CPU 上加载
    model.load_state_dict(torch.load('model.pth', map_location='cpu'))
    # CPU 训练的模型在 GPU 上加载
    model.load_state_dict(torch.load('model.pth', map_location='cuda:0'))

  加载到推理设备的最佳实践:
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

  ⚠️ PyTorch 2.6+ 推荐使用 weights_only=True 更安全
""")

print("=" * 60)
print("✅ 第 12 步完成！")
print("=" * 60)
