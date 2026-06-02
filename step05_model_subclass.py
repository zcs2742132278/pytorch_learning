"""
======================================================================
第 5 步：模型子类写法（nn.Module 深入）
======================================================================
学习目标：
  1. 深入理解 nn.Module 的运行机制
  2. 掌握 __init__ + forward 的标准范式
  3. 理解 nn.Sequential 的用法与限制
  4. 实现多分支结构（残差连接、多输出）
  5. 掌握 named_parameters()、named_modules()、apply() 等工具

关键词：nn.Module, forward, Sequential, 参数管理, 模型组合
======================================================================

PyTorch 中 99% 的自定义模型都是继承 nn.Module。
核心规则就两条：
  1. __init__() 中定义子层（如 nn.Linear, nn.Conv2d）
  2. forward() 中定义前向计算逻辑
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import set_seed

set_seed(42)

# ============================================================
print("=" * 60)
print("5.1 基础子类写法：重写 __init__ 和 forward")
print("=" * 60)


class MLP(nn.Module):
    """标准 MLP 模型"""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()  # ⚠️ 必须调用父类 __init__
        # 定义子层
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(0.3)
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        # 定义前向计算
        x = F.relu(self.fc1(x))
        x = self.bn(x)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = MLP(in_dim=10, hidden_dim=64, out_dim=3)
print(f"MLP 模型:")
print(model)
print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
print()

# 前向传播测试
x = torch.randn(4, 10)  # batch=4, features=10
y = model(x)
print(f"输入: {x.shape} → 输出: {y.shape}")
print(f"输出内容:\n{y}")
print()

# ============================================================
print("=" * 60)
print("5.2 nn.Sequential — 顺序模型的快捷写法")
print("=" * 60)

# 写法1: 传入有序字典
model_seq1 = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 3),
)
print("Sequential 写法1:")
print(model_seq1)

# 写法2: 使用 OrderedDict（可命名各层）
from collections import OrderedDict

model_seq2 = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(10, 64)),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(64, 32)),
    ('relu2', nn.ReLU()),
    ('fc3', nn.Linear(32, 3)),
]))
print("\nSequential 写法2 (OrderedDict):")
print(model_seq2)
print(f"按名称访问: {model_seq2.fc1}")

# Sequential 的局限性
print("\n⚠️ nn.Sequential 的局限:")
print("  - 只能逐层顺序执行")
print("  - 不支持多输入/多输出")
print("  - 不支持跳跃连接（残差网络）")
print("  - 不能有分支逻辑")
print("  ✅ 简单场景推荐用 Sequential，复杂场景用子类写法")

# ============================================================
print("\n" + "=" * 60)
print("5.3 参数管理")
print("=" * 60)

print("--- named_parameters() ---")
for name, param in model.named_parameters():
    print(f"  {name:25s}  shape={str(param.shape):15s}  requires_grad={param.requires_grad}")

print("\n--- named_modules() ---")
for name, module in model.named_modules():
    print(f"  {name:25s}  {module.__class__.__name__}")

print("\n--- 参数初始化 ---")
# nn.Linear 默认使用 Kaiming Uniform 初始化
# 可以手动修改

# 方式1: 直接在 __init__ 中初始化
class MLP_custom_init(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

        # 自定义初始化
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

m = MLP_custom_init(10, 64, 3)
print(f"fc1.weight 均值: {m.fc1.weight.mean().item():.6f}")

# 方式2: 使用 apply() 对每个子模块应用函数
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

model.apply(init_weights)  # 递归遍历所有子模块
print("apply() 对整体模型应用初始化完成")
print()

# ============================================================
print("=" * 60)
print("5.4 高级用法：跳跃连接（残差块）")
print("=" * 60)


class ResidualBlock(nn.Module):
    """残差块: output = input + F(input)"""
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        identity = x                    # 保存输入
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out = out + identity            # 跳跃连接
        return F.relu(out)


class ResNet_MLP(nn.Module):
    """由残差块堆叠的 MLP"""
    def __init__(self, in_dim, hidden_dim, out_dim, num_blocks=3):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(hidden_dim) for _ in range(num_blocks)
        ])
        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.input_proj(x))
        x = self.res_blocks(x)
        x = self.output_proj(x)
        return x


resnet = ResNet_MLP(in_dim=10, hidden_dim=64, out_dim=3)
x = torch.randn(4, 10)
y = resnet(x)
print(f"ResNet_MLP: 输入{x.shape} → 输出{y.shape}")
print(f"参数量: {sum(p.numel() for p in resnet.parameters()):,}")
print()

# ============================================================
print("=" * 60)
print("5.5 多输出模型")
print("=" * 60)


class MultiHeadModel(nn.Module):
    """共享 Backbone + 多个任务头"""
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        # 共享特征提取器
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # 多个输出头
        self.classifier_head = nn.Linear(hidden_dim, 10)  # 分类任务
        self.regression_head = nn.Linear(hidden_dim, 1)   # 回归任务

    def forward(self, x):
        features = self.backbone(x)
        class_out = self.classifier_head(features)
        reg_out = self.regression_head(features)
        return class_out, reg_out  # 返回两个输出


multi_model = MultiHeadModel(10, 64)
x = torch.randn(4, 10)
cls_out, reg_out = multi_model(x)
print(f"MultiHeadModel: 输入{x.shape}")
print(f"  分类输出: {cls_out.shape}")
print(f"  回归输出: {reg_out.shape}")
print()

# ============================================================
print("=" * 60)
print("5.6 实用技巧")
print("=" * 60)

print("--- 模式切换 ---")
model.train()   # 训练模式（Dropout/BN 生效）
model.eval()    # 评估模式（Dropout/BN 固定）

print(f"model.training = {model.training}  # 当前是训练模式")

print("\n--- 冻结/解冻参数 ---")
for name, param in model.named_parameters():
    if 'fc1' in name:
        param.requires_grad = False
        print(f"冻结: {name}")
    else:
        print(f"训练: {name}")

# 查看可训练参数
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"\n可训练参数: {trainable:,} / {total:,}")

print("\n--- 模型设备转移 ---")
print(f"当前设备: next(model.parameters()).device")
print("model.to('cuda') 可转移到 GPU（需要 CUDA）")

print("\n--- 获取中间层输出（Hook）---")
activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# 注册 hook
model.fc1.register_forward_hook(get_activation('fc1'))
x = torch.randn(1, 10)
model(x)
print(f"fc1 输出: shape={activations['fc1'].shape}")
print(f"fc1 输出值: {activations['fc1']}")

print("\n" + "=" * 60)
print("✅ 第 5 步完成！")
print("=" * 60)
