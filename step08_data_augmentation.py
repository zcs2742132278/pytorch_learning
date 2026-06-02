"""
======================================================================
第 8 步：数据增强（Data Augmentation）
======================================================================
学习目标：
  1. 理解数据增强的原理与必要性
  2. 掌握 torchvision.transforms 的使用
  3. 掌握自定义 transform 的方法
  4. 理解训练时的数据增强 vs 测试时的固定预处理
  5. 可视化增强效果

关键词：transforms.Compose, RandomHorizontalFlip, Normalize, 自定义 transform
======================================================================

为什么需要数据增强？
  - 扩大有效数据集大小（不增加标注成本）
  - 提高模型泛化能力（见过更多变体）
  - 减少过拟合
  - 模拟真实场景中的变化（光照、角度、遮挡等）

核心原则：增强后的样本对任务语义不变
  猫左右翻转 → 还是猫 ✅
  猫上下翻转 → 不太像猫了 ❌（取决于任务）
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils import set_seed

set_seed(42)

# ============================================================
print("=" * 60)
print("8.1 数据增强的原理")
print("=" * 60)

# 生成一些模拟的图像数据（28×28 手写数字风格）
def make_mock_image():
    """生成一个模拟的简单图形"""
    img = np.zeros((28, 28), dtype=np.float32)
    # 画一个圆
    for i in range(28):
        for j in range(28):
            if (i - 14) ** 2 + (j - 14) ** 2 < 81:
                img[i, j] = 1.0
    # 加一点噪声
    img += np.random.randn(28, 28) * 0.05
    return np.clip(img, 0, 1)

# 生成图像列表
images = [make_mock_image() for _ in range(4)]

# ============================================================
print("=" * 60)
print("8.2 torchvision.transforms 常用操作")
print("=" * 60)

# transforms 是 callable 对象，输入 PIL Image / Tensor，输出转换后的结果

# 转为 Tensor
to_tensor = transforms.ToTensor()  # H×W×C (numpy/PIL) → C×H×W (tensor), 值域 [0,1]

# 图像空间增强
augmentations = {
    "原图": lambda x: x,
    "随机水平翻转": transforms.RandomHorizontalFlip(p=1.0),
    "随机垂直翻转": transforms.RandomVerticalFlip(p=1.0),
    "随机旋转 ±30°": transforms.RandomRotation(degrees=30),
    "随机裁剪 + 缩放": transforms.RandomResizedCrop(size=28, scale=(0.6, 1.0)),
    "颜色抖动": transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    "随机灰度": transforms.RandomGrayscale(p=1.0),
    "高斯模糊": transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 2.0)),
}

print("常用 transforms 预览（将在可视化中展示）:")
for name in augmentations:
    print(f"  - {name}")

# ============================================================
print("\n" + "=" * 60)
print("8.3 构建增强 Pipeline（Compose）")
print("=" * 60)

# 训练时的增强 pipeline
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet 统计量
                         std=[0.229, 0.224, 0.225]),
])

# 测试时只需 ToTensor + Normalize（不做随机增强）
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

print("训练 Transform:")
print(train_transform)

print("\n测试 Transform:")
print(test_transform)

print("\n⚠️ 关键区别:")
print("  训练: 随机增强（提高泛化）")
print("  测试: 固定预处理（保证一致性）")
print()

# ============================================================
print("=" * 60)
print("8.4 自定义 Transform")
print("=" * 60)


class AddGaussianNoise:
    """自定义增加高斯噪声的 Transform"""
    def __init__(self, mean=0., std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return f"AddGaussianNoise(mean={self.mean}, std={self.std})"


class RandomErasing_custom:
    """自定义随机遮挡"""
    def __init__(self, p=0.5, scale=(0.02, 0.1)):
        self.p = p
        self.scale = scale

    def __call__(self, tensor):
        if torch.rand(1) > self.p:
            return tensor

        c, h, w = tensor.shape
        area = h * w
        # 随机遮挡区域大小
        erase_area = area * float(torch.empty(1).uniform_(*self.scale))
        aspect_ratio = float(torch.empty(1).uniform_(0.3, 3.3))

        erase_h = int(np.sqrt(erase_area * aspect_ratio))
        erase_w = int(np.sqrt(erase_area / aspect_ratio))

        if erase_h >= h or erase_w >= w:
            return tensor

        i = torch.randint(0, h - erase_h + 1, (1,)).item()
        j = torch.randint(0, w - erase_w + 1, (1,)).item()

        tensor[:, i:i+erase_h, j:j+erase_w] = 0
        return tensor


print("自定义 Transform 已定义:")
print(f"  - {AddGaussianNoise()}")
print(f"  - {RandomErasing_custom()}")

# ============================================================
print("\n" + "=" * 60)
print("8.5 将增强集成到 Dataset 中")
print("=" * 60)


class AugmentedDataset(Dataset):
    """带数据增强的数据集"""
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]   # Tensor [C, H, W]
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)  # 在这里做增强

        return img, label


# 模拟数据
mock_data = torch.rand(100, 3, 28, 28)
mock_labels = torch.randint(0, 10, (100,))

# 训练集使用增强
train_ds = AugmentedDataset(mock_data, mock_labels, transform=train_transform)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

xb, yb = next(iter(train_loader))
print(f"增强后的 batch: x.shape={xb.shape} (已归一化到 ImageNet 统计量)")
print(f"  x 均值={xb.mean():.4f}, x 标准差={xb.std():.4f}")
print(f"  x 取值范围: [{xb.min():.3f}, {xb.max():.3f}]")
print()

# ============================================================
print("=" * 60)
print("8.6 数据增强可视化")
print("=" * 60)

# 用真实的随机图像来做可视化
from PIL import ImageDraw

def create_sample_image():
    """创建一个带形状的彩色测试图像"""
    img = Image.new('RGB', (64, 64), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # 画一个红色矩形
    draw.rectangle([10, 10, 54, 54], fill=(200, 50, 50))
    # 画一个蓝色圆形
    draw.ellipse([25, 20, 45, 48], fill=(50, 50, 200))
    return img

sample_img = create_sample_image()

# 定义可视化用的 transforms
vis_transforms = {
    "原图": transforms.Compose([]),
    "水平翻转": transforms.RandomHorizontalFlip(p=1.0),
    "旋转 ±30°": transforms.RandomRotation(30),
    "随机裁剪": transforms.RandomResizedCrop(64, scale=(0.5, 1.0)),
    "颜色抖动": transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
    "随机灰度": transforms.RandomGrayscale(p=1.0),
    "高斯模糊": transforms.GaussianBlur(kernel_size=5, sigma=2.0),
}

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
fig.suptitle("数据增强效果展示", fontsize=14)

for ax, (name, transform) in zip(axes.flat, vis_transforms.items()):
    augmented = transform(sample_img)
    ax.imshow(augmented)
    ax.set_title(name)
    ax.axis('off')

plt.tight_layout()
plt.savefig("data/data_augmentation_demo.png", dpi=150)
plt.close()
print("图片已保存到: data/data_augmentation_demo.png")

# ============================================================
print("\n" + "=" * 60)
print("8.7 数据增强最佳实践")
print("=" * 60)
print("""
  1. 增强组合 (Compose):
     训练: ToTensor → 增强 → Normalize
     测试: ToTensor → Normalize

  2. 增强强度要合理:
     - 太弱: 效果有限
     - 太强: 破坏语义，模型学到噪声
     - 建议: 从小增强开始，逐步加强

  3. 不同任务选择不同增强:
     - 自然图像分类: 翻转、旋转、颜色抖动、RandAugment
     - 文字识别 (OCR): 不要垂直翻转、不要大角度旋转
     - 医学图像: 弹性变形、旋转、对比度调整（谨慎）
     - 目标检测: 同时变换图像和 bbox
     - 分割: 同时变换图像和 mask

  4. 现代增强技术:
     - MixUp: 混合两张图像
     - CutMix: 切一块区域替换为另一张图像的对应区域
     - AutoAugment: 自动搜索最优增强策略
     - RandAugment: 随机选择增强操作

  5. 不要在验证/测试集上做随机增强！
     只在训练集上做增强，保持测试时的确定性
""")

print("=" * 60)
print("✅ 第 8 步完成！")
print("=" * 60)
