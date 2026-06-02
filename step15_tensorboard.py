"""
======================================================================
第 15 步：TensorBoard 使用
======================================================================
学习目标：
  1. 掌握 SummaryWriter 的基本用法
  2. 学会记录标量、图像、模型图、直方图等
  3. 使用 TensorBoard 可视化训练过程
  4. 学会在 PyCharm 中启动 TensorBoard
  5. 对比不同实验的指标

关键词：SummaryWriter, add_scalar, add_image, add_graph, add_histogram
======================================================================

TensorBoard 是深度学习实验管理的标准工具:
  - 实时监控训练指标（loss, accuracy 等）
  - 可视化模型结构图
  - 比较不同实验
  - 查看权重分布、梯度分布
  - 嵌入（Embedding）可视化
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
from pathlib import Path
from utils import set_seed

set_seed(42)

# ============================================================
print("=" * 60)
print("15.1 TensorBoard 基础：SummaryWriter")
print("=" * 60)

# 创建 SummaryWriter
# 每次运行会创建一个带时间戳的子目录
log_dir = Path("data/runs/experiment_01")
writer = SummaryWriter(log_dir=str(log_dir))

print(f"TensorBoard 日志目录: {log_dir}")
print(f"启动方式:")
print(f"  终端:  tensorboard --logdir={log_dir.parent}")
print(f"  PyCharm: 右键 runs 目录 → Open in Terminal → 执行上述命令")
print(f"  或者: View → Tool Windows → TensorBoard → 选择 {log_dir.parent}")
print()

# ============================================================
print("=" * 60)
print("15.2 记录标量 (Scalar) — 最常用")
print("=" * 60)

# 准备数据
x = torch.randn(500, 20)
y = (x[:, 0] * 2 + x[:, 1] * 3 + x[:, 2] * -1 + torch.randn(500) * 0.5) > 0
y = y.long()

train_ds = TensorDataset(x[:400], y[:400])
val_ds = TensorDataset(x[400:], y[400:])
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

# 模型
model = nn.Sequential(
    nn.Linear(20, 128), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(64, 2),
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("训练中记录标量到 TensorBoard...")
for epoch in range(30):
    # 训练
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

    train_loss_avg = train_loss / train_total
    train_acc = train_correct / train_total

    # 验证
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb)
            loss = criterion(logits, yb)
            val_loss += loss.item() * len(yb)
            val_correct += (logits.argmax(1) == yb).sum().item()
            val_total += len(yb)

    val_loss_avg = val_loss / val_total
    val_acc = val_correct / val_total

    # ---- 记录到 TensorBoard ----
    writer.add_scalar('Loss/Train', train_loss_avg, epoch)
    writer.add_scalar('Loss/Val', val_loss_avg, epoch)
    writer.add_scalar('Accuracy/Train', train_acc, epoch)
    writer.add_scalar('Accuracy/Val', val_acc, epoch)
    writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

    if (epoch + 1) % 10 == 0:
        print(f"  epoch {epoch+1:2d}: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")

print("✅ 标量记录完成")
print()

# ============================================================
print("=" * 60)
print("15.3 记录模型图 (Graph)")
print("=" * 60)

# 传入一个输入样本，TensorBoard 会记录完整计算图
dummy_input = torch.randn(1, 20)
writer.add_graph(model, dummy_input)
print("✅ 模型图已记录")
print("  在 TensorBoard 的 GRAPHS 标签页查看")
print()

# ============================================================
print("=" * 60)
print("15.4 记录图像 (Image)")
print("=" * 60)

# 记录一些"图像"（这里用随机数据模拟）
mock_images = torch.randn(8, 3, 32, 32)  # 8 张 3×32×32

# add_images 接受 (N, C, H, W) 的 Tensor
writer.add_images('Sample/MockImages', mock_images, global_step=0)

# 也可以记录 matplotlib figure
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(np.sin(np.linspace(0, 4*np.pi, 100)))
ax.set_title("Sine Wave")
ax.set_xlabel("x")
ax.set_ylabel("sin(x)")

writer.add_figure('Sample/Matplotlib_Figure', fig, global_step=0)
plt.close()
print("✅ 图像已记录")
print()

# ============================================================
print("=" * 60)
print("15.5 记录直方图 (Histogram) — 监控权重分布")
print("=" * 60)

# 记录模型权重的分布（帮助检测梯度消失/爆炸等问题）
for name, param in model.named_parameters():
    writer.add_histogram(f'Parameters/{name}', param.data, global_step=0)
    if param.grad is not None:
        writer.add_histogram(f'Gradients/{name}', param.grad, global_step=0)

print("✅ 参数直方图已记录")
print()

# ============================================================
print("=" * 60)
print("15.6 记录嵌入 (Embedding) — 高维数据可视化")
print("=" * 60)

# 模拟一些嵌入向量和对应的标签
embeddings = torch.randn(100, 128)  # 100 个 128-d 嵌入向量
labels = torch.randint(0, 5, (100,))  # 5 类标签
# 模拟存储的图像（如果要可视化的话）
label_img = torch.randn(100, 3, 32, 32)  # 实际使用时放缩略图

writer.add_embedding(embeddings, metadata=labels, label_img=label_img,
                     global_step=0, tag='Sample/Embeddings')
print("✅ 嵌入已记录（可在 PROJECTOR 标签页查看）")
print()

# ============================================================
print("=" * 60)
print("15.7 记录 PR 曲线")
print("=" * 60)

# 用于二分类问题的 Precision-Recall 曲线
# 模拟预测分数和真实标签
num_samples = 100
mock_scores = torch.rand(num_samples)  # 预测为正类的概率
mock_labels = torch.randint(0, 2, (num_samples,))  # 真实标签

writer.add_pr_curve('Sample/PR_Curve', mock_labels, mock_scores, global_step=0)
print("✅ PR 曲线已记录")
print()

# ============================================================
print("=" * 60)
print("15.8 多实验对比")
print("=" * 60)

# 实验 1: lr=0.01
writer1 = SummaryWriter(log_dir="data/runs/exp_lr001")
# 实验 2: lr=0.001
writer2 = SummaryWriter(log_dir="data/runs/exp_lr0001")
# 实验 3: lr=0.0001
writer3 = SummaryWriter(log_dir="data/runs/exp_lr00001")

print("模拟 3 个不同学习率的实验...")
for exp_name, writer_exp, lr in [("lr=0.01", writer1, 0.01),
                                  ("lr=0.001", writer2, 0.001),
                                  ("lr=0.0001", writer3, 0.0001)]:
    model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 2))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(30):
        for xb, yb in train_loader:
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 只记录验证准确率（简化）
        model.eval()
        with torch.no_grad():
            correct = sum((model(xb).argmax(1) == yb).sum().item() for xb, yb in val_loader)
            acc = correct / len(val_ds)

        # hparam + metric 超参数对比
        writer_exp.add_scalar('Accuracy/Val', acc, epoch)

    print(f"  {exp_name}: 最终 val_acc={acc:.3f}")

# 超参数对比表
writer1.add_hparams(
    {'lr': 0.01, 'batch_size': 32, 'optimizer': 'SGD'},
    {'hparam/accuracy': acc},
)

writer2.add_hparams(
    {'lr': 0.001, 'batch_size': 32, 'optimizer': 'SGD'},
    {'hparam/accuracy': acc},
)

writer3.add_hparams(
    {'lr': 0.0001, 'batch_size': 32, 'optimizer': 'SGD'},
    {'hparam/accuracy': acc},
)

print(f"\n✅ 多实验对比完成")
print(f"  在 TensorBoard 中可以看到 3 条不同颜色的曲线")
print(f"  还可以用 HPARAMS 标签页对比超参数")

# 关闭所有 writer
writer.close()
writer1.close()
writer2.close()
writer3.close()

# ============================================================
print(f"\n{'='*60}")
print("15.9 PyCharm 中启动 TensorBoard")
print("=" * 60)
print("""
  方法1 (终端):
    cd D:/code/pytorch学习
    tensorboard --logdir=data/runs --port=6006

  方法2 (PyCharm 内置):
    View → Tool Windows → TensorBoard
    点击 + 号 → 选择 data/runs 目录

  方法3 (代码中启动):
    from torch.utils.tensorboard import SummaryWriter
    # 在命令行执行:
    # tensorboard --logdir=data/runs

  打开浏览器访问: http://localhost:6006

  TensorBoard 标签页:
  - SCALARS:     损失、准确率曲线
  - GRAPHS:      模型结构图
  - IMAGES:      记录的图片
  - HISTOGRAMS:  权重/梯度分布
  - PROJECTOR:   嵌入向量可视化
  - HPARAMS:     超参数对比
  - TEXT:        文本日志
  - PR CURVES:   精确率-召回率曲线
""")

# ============================================================
print("=" * 60)
print("15.10 SummaryWriter API 速查")
print("=" * 60)
print("""
  writer = SummaryWriter("runs/exp_name")

  # 记录标量（最常用）
  writer.add_scalar('Loss/Train', value, global_step)
  writer.add_scalars('Loss', {'train': t, 'val': v}, step)

  # 记录图像
  writer.add_image('Input', img_tensor, step)      # 单张 (C,H,W)
  writer.add_images('Batch', img_tensor, step)     # 批量 (N,C,H,W)
  writer.add_figure('Plot', matplotlib_figure, step)

  # 记录模型
  writer.add_graph(model, dummy_input)

  # 记录分布
  writer.add_histogram('Weights', tensor, step)

  # 记录文本
  writer.add_text('Config', str_config, step)

  # 记录音频
  writer.add_audio('Audio', audio_tensor, step, sample_rate=16000)

  # 记录超参数
  writer.add_hparams({'lr': 0.01, 'bs': 64}, {'acc': 0.95})

  # 记录嵌入
  writer.add_embedding(embeddings, metadata=labels)

  # 记录 PR 曲线
  writer.add_pr_curve('PR', labels, predictions, step)

  # 强制写入磁盘（否则可能在 buffer 中）
  writer.flush()

  # 关闭
  writer.close()
""")

print("=" * 60)
print("15.11 学习总结")
print("=" * 60)
print("""
  ┌──────┬────────────────────────────┬──────────────────────┐
  │ 步骤 │ 内容                        │ 核心技能              │
  ├──────┼────────────────────────────┼──────────────────────┤
  │ 1    │ 张量基础                    │ 数据操作基础           │
  │ 2    │ 自动微分                    │ 理解反向传播           │
  │ 3    │ 线性回归（原始+封装）        │ 训练循环 + 框架理解    │
  │ 4    │ 分类                        │ Sigmoid/Softmax/CE   │
  │ 5    │ 模型子类                    │ nn.Module 核心范式    │
  │ 6    │ Dataset                     │ 数据加载解耦           │
  │ 7    │ DataLoader                  │ 批量训练 + 性能        │
  │ 8    │ 数据增强                    │ 泛化能力 + transforms  │
  │ 9    │ 代码校验 + 封装             │ Trainer + EarlyStop   │
  │ 10   │ CNN 手写数字识别            │ 综合项目 1（CV入门）   │
  │ 11   │ 学习率衰减                  │ Scheduler + Warmup    │
  │ 12   │ 模型保存                    │ Checkpoint + 部署     │
  │ 13   │ 天气分类                    │ 综合项目 2（真实场景）  │
  │ 14   │ 迁移学习                    │ 预训练模型 + 微调      │
  │ 15   │ TensorBoard                 │ 实验管理 + 可视化      │
  └──────┴────────────────────────────┴──────────────────────┘

  你现在已经掌握的 PyTorch 核心能力:
  ✅ 能独立完成中小规模深度学习项目
  ✅ 能构建自定义模型、数据集和训练流程
  ✅ 能使用预训练模型进行迁移学习
  ✅ 能用 TensorBoard 管理实验
  ✅ 能保存/加载模型，部署到生产环境
  ✅ 对应岗位: 实习/初级深度学习工程师

  下一步建议:
  - 目标检测 (YOLO, Faster R-CNN)
  - 图像分割 (U-Net, DeepLab)
  - NLP (Transformer, BERT)
  - 生成模型 (GAN, VAE, Diffusion)
  - 模型部署 (ONNX, TensorRT, TorchServe)
  - 分布式训练 (DDP, FSDP)
""")

print("=" * 60)
print("✅ 15 步 PyTorch 学习之旅完成！恭喜！🎉")
print("=" * 60)
