"""
======================================================================
第 14 步：迁移学习（Transfer Learning）
======================================================================
学习目标：
  1. 理解迁移学习的核心思想（复用预训练模型的知识）
  2. 掌握两种迁移学习策略：微调（Fine-tuning）和特征提取（Feature Extraction）
  3. 使用 torchvision.models 加载预训练模型
  4. 修改分类头以适应新任务
  5. 掌握冻结/解冻参数和分层学习率

关键词：transfer learning, fine-tuning, pretrained, ResNet, freeze parameters
======================================================================

为什么迁移学习有效？
  - 预训练模型在大数据集（如 ImageNet 1400万张图）上学会了通用特征
  - 底层卷积学到边缘/纹理/形状等通用视觉特征
  - 即使新任务数据少（几百张），也能训出好模型

两种策略:
  1. 特征提取: 冻结预训练 backbone，只训练新分类头（数据少时用）
  2. 微调: 解冻全部参数，用更小 lr 调整（数据较多时用）
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import time
import copy
from utils import set_seed

set_seed(42)

# ============================================================
print("=" * 60)
print("14.1 迁移学习核心概念")
print("=" * 60)
print("""
  ImageNet (1400万 图片, 1000 类) 预训练 → 你的任务 (几百张, 10 类)

  ┌─────────────────────────────────────────────────────────┐
  │  预训练 ResNet-18 (ImageNet, 1000类)                    │
  │  ┌──────────┐  ┌──────────┐      ┌──────────┐          │
  │  │ Conv1    │→│ Layer1-4 │→ ... →│ fc (1000)│          │
  │  │ (通用特征)│  │ (通用特征)│      │ (分类头)  │          │
  │  └──────────┘  └──────────┘      └──────────┘          │
  │       ↓ 复用的部分        ↑ 替换这部分                   │
  │  ┌──────────────────┐  ┌──────────┐                    │
  │  │ 冻结 / 低 lr微调  │→│ fc (10)  │  ← 随机初始化训练    │
  │  └──────────────────┘  └──────────┘                    │
  └─────────────────────────────────────────────────────────┘
""")

# ============================================================
print("=" * 60)
print("14.2 加载预训练模型")
print("=" * 60)

# 列出可用的预训练模型
print("torchvision 中的预训练模型（部分）:")
available_models = [
    ('resnet18', 'ResNet-18', '11.7M'),
    ('resnet34', 'ResNet-34', '21.8M'),
    ('resnet50', 'ResNet-50', '25.6M'),
    ('efficientnet_b0', 'EfficientNet-B0', '5.3M'),
    ('mobilenet_v2', 'MobileNetV2', '3.5M'),
    ('densenet121', 'DenseNet-121', '8.0M'),
    ('convnext_tiny', 'ConvNeXt-Tiny', '28.6M'),
]
for key, name, params in available_models:
    print(f"  {key:20s} → {name:18s} ({params} params)")

# 加载 ResNet-18（ImageNet 预训练）
print(f"\n加载 ResNet-18 (ImageNet 预训练):")
try:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    print(f"  模型加载成功!")
    print(f"  原始分类头: {model.fc}")
    print(f"  输出类别: 1000 (ImageNet)")
except Exception as e:
    print(f"  加载失败（可能网络问题）: {e}")
    print("  使用随机初始化的模型进行演示...")
    model = models.resnet18(weights=None)

print()

# ============================================================
print("=" * 60)
print("14.3 数据准备（CIFAR-10 → 适配到 224×224）")
print("=" * 60)

# 预训练模型需要 224×224 输入（ImageNet 标准尺寸）
# 对于 CIFAR-10 (32×32)，需要上采样
train_transform = transforms.Compose([
    transforms.Resize(224),                    # 上采样到 224×224
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet 统计量
                         [0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# 加载数据
data_dir = "./data"
train_dataset = torchvision.datasets.CIFAR10(
    root=data_dir, train=True, transform=train_transform, download=True
)
test_dataset = torchvision.datasets.CIFAR10(
    root=data_dir, train=False, transform=test_transform, download=True
)

# 划分
train_size = int(0.85 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(val_subset, batch_size=64, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

print(f"训练集: {len(train_subset):,} | 验证集: {len(val_subset):,} | 测试集: {len(test_dataset):,}")
print()

# ============================================================
print("=" * 60)
print("14.4 策略 A: 特征提取（冻结 Backbone）")
print("=" * 60)


def build_feature_extractor(base_model, num_classes=10):
    """特征提取器：冻结 backbone，只训练新的分类头"""
    model = copy.deepcopy(base_model)

    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    # 替换最后一层全连接
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes),
    )

    # 只有新的 fc 层可训练
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    return model


# ============================================================
print("=" * 60)
print("14.5 策略 B: 微调（全模型 + 分层学习率）")
print("=" * 60)


def build_finetune_model(base_model, num_classes=10):
    """微调模型：所有参数可训练，但分类头学习率更大"""
    model = copy.deepcopy(base_model)

    # 所有参数可训练
    for param in model.parameters():
        param.requires_grad = True

    # 替换分类头
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes),
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    return model


# ============================================================
print("=" * 60)
print("14.6 训练函数")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


def train_model(model, train_loader, val_loader, epochs=15,
                lr=0.001, name="Model"):
    """通用训练函数"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    best_model_state = None
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
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

        scheduler.step()

        t_acc = train_correct / train_total
        v_acc = val_correct / val_total
        history['train_loss'].append(train_loss / train_total)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(val_loss / val_total)
        history['val_acc'].append(v_acc)

        if v_acc > best_acc:
            best_acc = v_acc
            best_model_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 5 == 0:
            print(f"  [{name}] epoch {epoch+1:2d}/{epochs}: "
                  f"train_acc={t_acc:.3f}, val_acc={v_acc:.3f}")

    # 恢复最佳
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, best_acc, history


# ============================================================
print("=" * 60)
print("14.7 对比实验")
print("=" * 60)

base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 策略 A: 特征提取
print("\n--- 策略 A: 特征提取 ---")
model_fe = build_feature_extractor(base_model, num_classes=10)
model_fe, acc_fe, hist_fe = train_model(
    model_fe, train_loader, val_loader, epochs=10, lr=0.001, name="FeatureExtract"
)
print(f"  最佳验证准确率: {acc_fe:.4f}")

# 策略 B: 微调（用更小的 lr）
print("\n--- 策略 B: 微调（全部参数，较小 lr）---")
model_ft = build_finetune_model(base_model, num_classes=10)
model_ft, acc_ft, hist_ft = train_model(
    model_ft, train_loader, val_loader, epochs=10, lr=0.0001, name="FineTune"
)
print(f"  最佳验证准确率: {acc_ft:.4f}")

# ============================================================
print(f"\n{'='*60}")
print("14.8 对比可视化")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f"迁移学习策略对比 (CIFAR-10)", fontsize=14)

for name, hist, ax in [("Feature Extract", hist_fe, axes[0]), ("Fine-tune", hist_ft, axes[1])]:
    ax.plot(hist['train_acc'], label='Train Acc')
    ax.plot(hist['val_acc'], label='Val Acc')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title(f"{name} (Best Val: {max(hist['val_acc']):.4f})")
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig("data/transfer_learning_comparison.png", dpi=150)
plt.close()
print("图片已保存到: data/transfer_learning_comparison.png")

print(f"\n最终对比:")
print(f"  特征提取 最佳验证准确率: {acc_fe:.4f}")
print(f"  微调     最佳验证准确率: {acc_ft:.4f}")

# ============================================================
print(f"\n{'='*60}")
print("14.9 迁移学习实战指南")
print("=" * 60)
print("""
  数据量 → 策略选择:

  ┌──────────────────┬─────────────────────────────────────┐
  │ 数据量            │ 推荐策略                             │
  ├──────────────────┼─────────────────────────────────────┤
  │ < 1,000 张       │ 特征提取（冻结全部 backbone）          │
  │ 1,000 - 10,000   │ 微调最后 1-2 层                      │
  │ 10,000 - 100,000 │ 微调全部（小 lr: 1e-4 ~ 1e-5）       │
  │ > 100,000        │ 从头训练 + 预训练对比                  │
  └──────────────────┴─────────────────────────────────────┘

  分层学习率（高级技巧）:
  optimizer = optim.SGD([
      {'params': model.conv1.parameters(), 'lr': 1e-5},     # 底层: 极小 lr
      {'params': model.layer1.parameters(), 'lr': 1e-5},
      {'params': model.layer2.parameters(), 'lr': 1e-4},    # 中层: 较小 lr
      {'params': model.layer3.parameters(), 'lr': 1e-4},
      {'params': model.layer4.parameters(), 'lr': 1e-3},    # 高层: 较大 lr
      {'params': model.fc.parameters(), 'lr': 1e-3},        # 分类头: 最大 lr
  ], lr=1e-4, momentum=0.9)

  常见预训练模型的选择:
  - 追求精度: ResNet-50/101, ConvNeXt, ViT
  - 速度优先: MobileNetV2/V3, EfficientNet, ShuffleNet
  - 平衡: ResNet-18/34, EfficientNet-B0
  - 目标检测: ResNet-50 + FPN
  - 分割: ResNet/ConvNeXt + DeepLabV3
""")

# ============================================================
print("=" * 60)
print("14.10 测试集评估（特征提取模型）")
print("=" * 60)

model_fe.eval()
test_correct, test_total = 0, 0
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model_fe(xb)
        test_correct += (logits.argmax(1) == yb).sum().item()
        test_total += len(yb)

print(f"特征提取模型 测试集准确率: {test_correct/test_total:.4f}")
print()

print("=" * 60)
print("✅ 第 14 步完成！")
print("=" * 60)
