"""
======================================================================
第 1 步：张量（Tensor）与标量（Scalar）
======================================================================
学习目标：
  1. 理解张量是 PyTorch 的核心数据结构
  2. 掌握张量的创建、属性查询、类型转换
  3. 掌握张量的基本运算（索引、切片、变形、拼接）
  4. 理解标量（0 维张量）的概念
  5. 掌握 CPU ↔ GPU 张量转移

关键词：torch.tensor, shape, dtype, device, reshape, cat, stack
======================================================================
"""

import torch
import numpy as np
from utils import set_seed, print_tensor_info, get_device

set_seed(42)

print("=" * 60)
print("1.1 张量的创建")
print("=" * 60)

# ---- 从 Python 列表创建 ----
a = torch.tensor([1, 2, 3])
b = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

print("从列表创建 1D 张量:")
print_tensor_info(a, "a")

print("从列表创建 2D 张量:")
print_tensor_info(b, "b")

# ---- 使用工厂方法创建 ----
zeros = torch.zeros(3, 4)          # 全 0
ones = torch.ones(2, 3)            # 全 1
rand = torch.rand(2, 2)            # [0, 1) 均匀分布
randn = torch.randn(2, 2)          # 标准正态分布 N(0,1)
arange = torch.arange(0, 10, 2)   # 等差数列
linspace = torch.linspace(0, 1, 5) # 等间距

print("常用工厂方法:")
print(f"  zeros(3,4)   shape={zeros.shape}")
print(f"  ones(2,3)    shape={ones.shape}")
print(f"  rand(2,2)   =\n{rand}")
print(f"  randn(2,2)  =\n{randn}")
print(f"  arange(0,10,2) = {arange}")
print(f"  linspace(0,1,5) = {linspace}")
print()

# ---- 从 NumPy 数组创建 ----
np_arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
from_np = torch.from_numpy(np_arr)  # 共享内存
from_np_copy = torch.tensor(np_arr) # 拷贝

print("从 NumPy 创建:")
print(f"  from_numpy (共享内存):  {from_np}")
print(f"  tensor (拷贝):          {from_np_copy}")
print()

# ---- 创建与已有张量相同属性的张量 ----
c = torch.randn(2, 3)
like_zeros = torch.zeros_like(c)
like_ones = torch.ones_like(c)
print(f"  zeros_like 保持 shape={like_zeros.shape}, dtype={like_zeros.dtype}")
print(f"  ones_like  保持 shape={like_ones.shape}, dtype={like_ones.dtype}")
print()

print("=" * 60)
print("1.2 张量属性")
print("=" * 60)

x = torch.randn(2, 3, 4)
print(f"shape:   {x.shape}     # 形状")
print(f"ndim:    {x.ndim}       # 维度数（秩）")
print(f"dtype:   {x.dtype}     # 数据类型")
print(f"device:  {x.device}    # 所在设备")
print(f"numel:   {x.numel()}   # 元素总数")
print(f"stride:  {x.stride()}  # 步长（各维度的内存跨度）")
print()

print("=" * 60)
print("1.3 索引与切片（与 NumPy 一致）")
print("=" * 60)

x = torch.arange(12).reshape(3, 4)
print(f"原始张量 (3×4):\n{x}\n")

print(f"x[0, 0]   = {x[0, 0]}       # 单个元素 → 标量")
print(f"x[0]      = {x[0]}           # 第 0 行")
print(f"x[:, 1]   = {x[:, 1]}        # 第 1 列")
print(f"x[:2, 1:] = \n{x[:2, 1:]}     # 前 2 行 + 后 3 列")
print(f"x[::2]    = \n{x[::2]}         # 每隔一行")
print(f"x[x > 5]  = {x[x > 5]}       # 布尔索引")

# 修改切片会影响原张量（共享内存）
y = x[:2, :2]
y[0, 0] = 999
print(f"\n修改切片 y (y[0,0]=999) 后，x 也被修改:")
print(f"x =\n{x}")
print()

print("=" * 60)
print("1.4 形状操作")
print("=" * 60)

x = torch.arange(12)
print(f"原始: {x}")

# reshape: -1 表示自动推导
print(f"reshape(3,4):   \n{x.reshape(3, 4)}")
print(f"reshape(2,-1):  \n{x.reshape(2, -1)}")

# view: 与 reshape 类似，但要求内存连续
print(f"view(3,4):      \n{x.view(3, 4)}")

# 添加/移除维度
print(f"unsqueeze(0).shape: {x.unsqueeze(0).shape}  # 在第 0 维前加一维")
print(f"unsqueeze(1).shape: {x.unsqueeze(1).shape}  # 在第 1 维前加一维")
print(f"squeeze().shape:    {x.unsqueeze(0).squeeze(0).shape}  # 移除大小为 1 的维度")

# 展平
print(f"flatten: {x.reshape(2, 2, 3).flatten()}")
print()

print("=" * 60)
print("1.5 拼接与堆叠")
print("=" * 60)

a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

print(f"a =\n{a}")
print(f"b =\n{b}")

# cat: 延已有维度拼接
print(f"\ntorch.cat([a,b], dim=0):\n{torch.cat([a, b], dim=0)}  # 纵向拼接")
print(f"\ntorch.cat([a,b], dim=1):\n{torch.cat([a, b], dim=1)}  # 横向拼接")

# stack: 新增一个维度
print(f"\ntorch.stack([a,b], dim=0):\n{torch.stack([a, b], dim=0)}  # shape=(2,2,2)")
print(f"torch.stack([a,b], dim=1):\n{torch.stack([a, b], dim=1)}  # shape=(2,2,2)")
print()

print("=" * 60)
print("1.6 张量运算")
print("=" * 60)

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(f"a = {a}")
print(f"b = {b}")
print(f"a + b      = {a + b}        # 逐元素加法")
print(f"a - b      = {a - b}        # 逐元素减法")
print(f"a * b      = {a * b}        # 逐元素乘法（非矩阵乘）")
print(f"a / b      = {a / b}        # 逐元素除法")
print(f"a @ b      = {a @ b}        # 点积（1D 向量内积）")
print(f"torch.dot(a,b) = {torch.dot(a, b)}  # 向量点积")
print(f"a ** 2     = {a ** 2}       # 逐元素幂")
print(f"a.sum()    = {a.sum()}      # 求和")
print(f"a.mean()   = {a.mean()}     # 均值")
print(f"a.max()    = {a.max()}      # 最大值")
print(f"a.argmax() = {a.argmax()}   # 最大值索引")
print()

# 广播机制
print("广播机制示例:")
x = torch.ones(3, 1)
y = torch.tensor([1.0, 2.0, 3.0, 4.0])
print(f"  x.shape = {x.shape}  (3×1)")
print(f"  y.shape = {y.shape}  (4,)")
print(f"  (x+y).shape = {(x + y).shape} 广播为 3×4")
print(f"  x + y =\n{x + y}")
print()

print("=" * 60)
print("1.7 标量（0 维张量）")
print("=" * 60)

scalar = torch.tensor(3.14)
print(f"标量: {scalar}")
print(f"  shape: {scalar.shape}   # 空列表，表示 0 维")
print(f"  item(): {scalar.item()} # 提取 Python 数值")

# 许多统计操作返回标量
loss = torch.randn(100).sum()
print(f"\nloss = torch.randn(100).sum()")
print(f"  loss:       {loss}")
print(f"  loss.shape: {loss.shape}")
print(f"  loss.item(): {loss.item():.4f}")
print()

print("=" * 60)
print("1.8 数据类型转换")
print("=" * 60)

x = torch.tensor([1, 2, 3])
print(f"int64:   {x}, dtype={x.dtype}")

x_float = x.float()       # → float32
x_double = x.double()     # → float64
x_half = x.half()         # → float16
x_long = x_float.long()   # → int64

print(f"float(): {x_float},  dtype={x_float.dtype}")
print(f"double():{x_double}, dtype={x_double.dtype}")
print(f"half():  {x_half},  dtype={x_half.dtype}")
print(f"long():  {x_long},  dtype={x_long.dtype}")

# type() 和 to() 方法
print(f"\nto(dtype=float32): {x.to(dtype=torch.float32)}")
print()

print("=" * 60)
print("1.9 设备转移")
print("=" * 60)

device = get_device()
print(f"当前设备: {device}")

x = torch.randn(3, 3)
print(f"创建在 CPU:     device={x.device}")

if torch.cuda.is_available():
    x_gpu = x.to("cuda")       # 或 x.cuda()
    print(f"转移到 GPU:     device={x_gpu.device}")
    x_cpu = x_gpu.to("cpu")    # 或 x_gpu.cpu()
    print(f"转移回 CPU:     device={x_cpu.device}")
else:
    print("（本机无 GPU，跳过 GPU 转移演示）")

print()
print("=" * 60)
print("✅ 第 1 步完成！")
print("=" * 60)
