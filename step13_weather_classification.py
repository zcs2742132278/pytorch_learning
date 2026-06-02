"""
======================================================================
第 13 步：天气分类 — 真实场景图像分类项目
======================================================================
学习目标：
  1. 从 MNIST 玩具数据集过渡到真实场景分类
  2. 使用 torchvision.datasets.ImageFolder 加载自定义图片数据集
  3. 构建更深的 CNN + 数据增强 pipeline
  4. 使用混淆矩阵分析模型表现
  5. 综合运用前 12 步的所有技术

关键词：ImageFolder, 混淆矩阵, 真实场景分类, 完整 pipeline
======================================================================

由于天气图片数据集需要下载，这里演示两种方案：
  A) 使用 torchvision 内置的 CIFAR-10 作为真实场景数据集
  B) 提供 ImageFolder 的标准接口（注释代码），可替换为天气数据集
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path
from utils import set_seed

set_seed(42)

# ============================================================
print("=" * 60)
print("13.1 数据集准备（CIFAR-10）")
print("=" * 60)
print("CIFAR-10 包含 10 类真实场景图片：")
print("  飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车")
print("  每张 32×32 彩色图，共 60,000 张")
print()

# 训练和测试的预处理
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),      # 先填充再随机裁剪
    transforms.RandomHorizontalFlip(p=0.5),     # 随机水平翻转
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),  # CIFAR-10 统计量
                         (0.2023, 0.1994, 0.2010)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

# 加载数据
data_dir = Path("./data")
train_dataset = torchvision.datasets.CIFAR10(
    root=data_dir, train=True, transform=train_transform, download=True
)
test_dataset = torchvision.datasets.CIFAR10(
    root=data_dir, train=False, transform=test_transform, download=True
)

# 类别名称
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# 划分训练/验证集
train_size = int(0.85 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

# DataLoader
train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=0)
val_loader = DataLoader(val_subset, batch_size=128, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

print(f"训练集: {len(train_subset):,} | 验证集: {len(val_subset):,} | 测试集: {len(test_dataset):,}")
print(f"图像大小: 3×32×32 | 类别数: 10")
print()

# ============================================================
print("=" * 60)
print("13.2 使用 ImageFolder 加载自定义数据集的接口")
print("=" * 60)
print("""
如果你的图片按以下目录结构组织:
  data/weather/
    ├── sunny/
    │   ├── img001.jpg
    │   ├── img002.jpg
    │   └── ...
    ├── rainy/
    │   ├── img001.jpg
    │   └── ...
    ├── cloudy/
    └── snowy/

则可以用一行代码加载:

  dataset = torchvision.datasets.ImageFolder(
      root='data/weather',
      transform=train_transform
  )

ImageFolder 会自动:
  - 用子目录名作为类别标签（按字母序）
  - 通过 dataset.classes 查看类别名
  - 通过 dataset.class_to_idx 查看类名→索引的映射
""")

# ============================================================
print("=" * 60)
print("13.3 构建深度 CNN")
print("=" * 60)


class ResidualBlock(nn.Module):
    """带残差连接的卷积块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 如果维度不匹配，用 1x1 卷积做投影
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 残差连接
        return F.relu(out)


class WeatherCNN(nn.Module):
    """CIFAR-10 分类用的 CNN（ResNet 风格）"""
    def __init__(self, num_classes=10):
        super().__init__()

        # 初始卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # 残差层
        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(64, 128, 3, stride=2)
        self.layer3 = self._make_layer(128, 256, 3, stride=2)

        # 全局平均池化 + 分类器
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def _make_layer(self, in_c, out_c, num_blocks, stride):
        layers = [ResidualBlock(in_c, out_c, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_c, out_c, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

model = WeatherCNN(num_classes=10).to(device)
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

# Shape flow
x_dummy = torch.randn(1, 3, 32, 32).to(device)
with torch.no_grad():
    y_dummy = model(x_dummy)
print(f"输入: (1, 3, 32, 32) → 输出: {y_dummy.shape}")
print()

# ============================================================
print("=" * 60)
print("13.4 训练（含学习率调度）")
print("=" * 60)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.001)

epochs = 50
train_losses, val_losses = [], []
train_accs, val_accs = [], []

print(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<10} {'Val Loss':<12} {'Val Acc':<10} {'LR':<12}")
print("-" * 64)

for epoch in range(epochs):
    epoch_start = time.time()

    # -- 训练 --
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

    # -- 验证 --
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

    scheduler.step()  # 更新学习率

    # 记录
    train_losses.append(train_loss / train_total)
    train_accs.append(train_correct / train_total)
    val_losses.append(val_loss / val_total)
    val_accs.append(val_correct / val_total)

    elapsed = time.time() - epoch_start
    current_lr = optimizer.param_groups[0]['lr']
    if (epoch + 1) % 10 == 0:
        print(f"{epoch+1:<8} {train_losses[-1]:<12.4f} {train_accs[-1]:<10.4f} "
              f"{val_losses[-1]:<12.4f} {val_accs[-1]:<10.4f} {current_lr:<12.6f}")

# ============================================================
print(f"\n{'='*60}")
print("13.5 测试集评估 + 混淆矩阵")
print("=" * 60)

model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        logits = model(xb)
        preds = logits.argmax(1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(yb.tolist())

test_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
print(f"测试集准确率: {test_acc:.4f}")

# 计算混淆矩阵
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(all_labels, all_preds)
cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)  # 按行归一化

print(f"\n分类报告:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# ============================================================
print("=" * 60)
print("13.6 可视化")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f"CIFAR-10 分类结果 (Test Acc: {test_acc:.4f})", fontsize=14)

# 损失曲线
axes[0].plot(train_losses, label='Train Loss')
axes[0].plot(val_losses, label='Val Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss Curves')
axes[0].legend()
axes[0].grid(True)

# 准确率曲线
axes[1].plot(train_accs, label='Train Acc')
axes[1].plot(val_accs, label='Val Acc')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy Curves')
axes[1].legend()
axes[1].grid(True)

# 混淆矩阵
im = axes[2].imshow(cm_normalized, cmap='Blues', vmin=0, vmax=1)
axes[2].set_xticks(range(10))
axes[2].set_yticks(range(10))
axes[2].set_xticklabels(class_names, rotation=45, ha='right', fontsize=7)
axes[2].set_yticklabels(class_names, fontsize=7)
axes[2].set_title('Confusion Matrix (Normalized)')
axes[2].set_xlabel('Predicted')
axes[2].set_ylabel('True')
plt.colorbar(im, ax=axes[2], fraction=0.046)

plt.tight_layout()
plt.savefig("data/weather_classification_results.png", dpi=150, bbox_inches='tight')
plt.close()
print("图片已保存到: data/weather_classification_results.png")

# ============================================================
print(f"\n{'='*60}")
print("13.7 使用 Python 内存优化技巧")
print("=" * 60)
print("""
  1. torch.cuda.empty_cache() - 清空 GPU 缓存
  2. del variable - 及时释放不再使用的变量
  3. pin_memory=True - DataLoader 中使用锁页内存
  4. torch.no_grad() - 推理时禁用梯度
  5. model.half() - 混合精度训练（FP16）
  6. gradient_accumulation - 梯度累积，模拟大 batch

  示例: 在循环结束时清理
  if torch.cuda.is_available():
      torch.cuda.empty_cache()
""")

# ============================================================
print("=" * 60)
print("13.8 真实天气数据集接口（供替换）")
print("=" * 60)
print("""
  # 获取天气数据集后，只需替换数据加载部分:

  train_transform = transforms.Compose([
      transforms.Resize((224, 224)),        # 天气图片通常更大
      transforms.RandomHorizontalFlip(),
      transforms.RandomRotation(15),
      transforms.ColorJitter(brightness=0.3, contrast=0.3),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225]),
  ])

  train_dataset = torchvision.datasets.ImageFolder(
      root='data/weather/train',
      transform=train_transform
  )

  # 其他代码完全相同! Dataset → DataLoader → Model → Train
  # 这就是解耦的好处
""")

print("=" * 60)
print("✅ 第 13 步完成！")
print("=" * 60)
