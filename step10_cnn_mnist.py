"""
======================================================================
第 10 步：CNN 手写数字识别（MNIST）
======================================================================
学习目标：
  1. 掌握卷积神经网络的组成（Conv2d + Pool + FC）
  2. 理解 CNN 中 shape 的变化（channels, spatial 维度）
  3. 使用 PyTorch 内置的 MNIST 数据集
  4. 综合运用前 9 步的所有知识

关键词：Conv2d, MaxPool2d, BatchNorm2d, Dropout, MNIST
======================================================================

CNN 架构模式:
  Input → [Conv → ReLU → (BN) → Pool] × N → Flatten → FC → Output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
from utils import set_seed

set_seed(42)

# ============================================================
print("=" * 60)
print("10.1 加载 MNIST 数据集")
print("=" * 60)

# 定义预处理
train_transform = transforms.Compose([
    transforms.RandomRotation(10),      # 数据增强：小角度旋转
    transforms.RandomAffine(0, translate=(0.1, 0.1)),  # 平移
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),  # MNIST 均值/标准差
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

# 下载并加载数据
train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, transform=train_transform, download=True
)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, transform=test_transform, download=True
)

# 划分验证集
from torch.utils.data import random_split
train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

# DataLoader
train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

print(f"MNIST 数据集:")
print(f"  训练集: {len(train_subset):,} 张")
print(f"  验证集: {len(val_subset):,} 张")
print(f"  测试集: {len(test_dataset):,} 张")
print(f"  图像大小: 28×28 (单通道)")
print(f"  类别数: 10 (0-9)")

# 查看一个 batch
xb, yb = next(iter(train_loader))
print(f"\n  batch shape: {xb.shape}  (N, C, H, W)")
print(f"  batch labels: {yb[:10].tolist()}")
print()

# ============================================================
print("=" * 60)
print("10.2 构建 CNN 模型")
print("=" * 60)


class CNN_MNIST(nn.Module):
    """经典 CNN 用于 MNIST"""
    def __init__(self, num_classes=10):
        super().__init__()

        # 卷积层 1: 1×28×28 → 32×14×14
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # 卷积层 2: 32×14×14 → 64×7×7
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # 卷积层 3: 64×7×7 → 128×3×3
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming 初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        return x

    def show_shape_flow(self, input_shape=(1, 1, 28, 28)):
        """打印各层 shape 变化（调试用）"""
        import torchsummary
        x = torch.randn(*input_shape)
        print(f"{'Layer':<20} {'Output Shape':<20} {'Param #':<15}")
        print("-" * 55)
        for name, layer in self.named_children():
            x = layer(x)
            params = sum(p.numel() for p in layer.parameters())
            print(f"{name:<20} {str(list(x.shape)):<20} {params:<15,}")


model = CNN_MNIST(num_classes=10)
print(f"CNN 模型:")
print(model)
print(f"\n总参数量: {sum(p.numel() for p in model.parameters()):,}")
print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# 展示 shape 变化
print("\n各层 shape 变化:")
model.show_shape_flow()
print()

# ============================================================
print("=" * 60)
print("10.3 训练 CNN")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

model = CNN_MNIST().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练
epochs = 10
train_losses, val_losses = [], []
train_accs, val_accs = [], []

print(f"\n{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<10} {'Val Loss':<12} {'Val Acc':<10} {'Time':<10}")
print("-" * 62)

for epoch in range(epochs):
    epoch_start = time.time()

    # 训练
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        logits = model(xb)
        loss = criterion(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * len(yb)
        train_correct += (logits.argmax(1) == yb).sum().item()
        train_total += len(yb)

    # 验证
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            val_loss += loss.item() * len(yb)
            val_correct += (logits.argmax(1) == yb).sum().item()
            val_total += len(yb)

    # 记录
    train_losses.append(train_loss / train_total)
    train_accs.append(train_correct / train_total)
    val_losses.append(val_loss / val_total)
    val_accs.append(val_correct / val_total)

    elapsed = time.time() - epoch_start
    print(f"{epoch+1:<8} {train_losses[-1]:<12.4f} {train_accs[-1]:<10.4f} "
          f"{val_losses[-1]:<12.4f} {val_accs[-1]:<10.4f} {elapsed:<10.2f}s")

# ============================================================
print(f"\n{'='*60}")
print("10.4 测试集评估")
print("=" * 60)

model.eval()
test_correct, test_total = 0, 0
all_preds, all_labels = [], []

with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        preds = logits.argmax(1)

        test_correct += (preds == yb).sum().item()
        test_total += len(yb)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(yb.cpu().tolist())

test_acc = test_correct / test_total
print(f"测试集准确率: {test_acc:.4f} ({test_correct}/{test_total})")

# ============================================================
print(f"\n{'='*60}")
print("10.5 结果可视化")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f"CNN MNIST 结果 (Test Acc: {test_acc:.4f})", fontsize=14)

# 损失曲线
axes[0, 0].plot(train_losses, label='Train Loss')
axes[0, 0].plot(val_losses, label='Val Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Loss Curves')
axes[0, 0].legend()
axes[0, 0].grid(True)

# 准确率曲线
axes[0, 1].plot(train_accs, label='Train Acc')
axes[0, 1].plot(val_accs, label='Val Acc')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].set_title('Accuracy Curves')
axes[0, 1].legend()
axes[0, 1].grid(True)
axes[0, 1].set_ylim([0.9, 1.0])

# 预测示例
sample_imgs, sample_labels = [], []
for xb, yb in test_loader:
    sample_imgs.append(xb[:10])
    sample_labels.append(yb[:10])
    break

sample_imgs = sample_imgs[0]
sample_labels = sample_labels[0]

model.eval()
with torch.no_grad():
    sample_preds = model(sample_imgs.to(device)).argmax(1).cpu()

for i in range(10):
    ax = axes[1, 0] if i < 5 else axes[1, 1]
    idx = i % 5
    # 反标准化
    img = sample_imgs[i, 0].cpu() * 0.3081 + 0.1307
    color = 'green' if sample_preds[i] == sample_labels[i] else 'red'
    ax_row = i % 5
    ax_col = i // 5
    target_ax = axes[1, i // 5]

    if idx == 0:
        target_ax.cla()

    target_ax.imshow(img.numpy(), cmap='gray_r')
    target_ax.set_title(f"True: {sample_labels[i]}, Pred: {sample_preds[i]}", color=color)
    target_ax.axis('off')

axes[1, 0].set_title('5 个测试样本预测')
axes[1, 1].set_title('更多测试样本预测')

plt.tight_layout()
plt.savefig("data/cnn_mnist_results.png", dpi=150)
plt.close()
print("图片已保存到: data/cnn_mnist_results.png")

# ============================================================
print(f"\n{'='*60}")
print("10.6 CNN 参数计算公式速查")
print("=" * 60)
print("""
  Conv2d 输出大小:
    H_out = (H_in + 2*P - K) / S + 1

  MaxPool2d 输出大小:
    H_out = (H_in - K) / S + 1

  参数量:
    Conv2d:  C_in × C_out × K × K + C_out (bias)
    Linear:  in_features × out_features + out_features (bias)
    BatchNorm2d: 2 × num_features (weight + bias)

  本例参数量计算:
    conv1: 1×32×3×3 + 32 = 320
    conv2: 32×64×3×3 + 64 = 18,496
    conv3: 64×128×3×3 + 128 = 73,856
    fc[1]: 128*3*3×256 + 256 = 295,168
    fc[3]: 256×10 + 10 = 2,570
    总计: ≈ 390,410
""")

print("=" * 60)
print("✅ 第 10 步完成！")
print("=" * 60)
