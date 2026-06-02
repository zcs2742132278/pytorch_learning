"""
======================================================================
第 2 步：自动微分（Autograd）
======================================================================
学习目标：
  1. 理解计算图（Computation Graph）的构建与反向传播
  2. 掌握 requires_grad 属性与 backward() 方法
  3. 理解梯度累加与清零机制
  4. 掌握 detach()、no_grad()、zero_grad() 的用法
  5. 理解叶子节点与非叶子节点的梯度规则

关键词：requires_grad, backward(), grad, detach, torch.no_grad, zero_grad
======================================================================
"""

import torch
from utils import set_seed

set_seed(42)

print("=" * 60)
print("2.1 基础自动微分")
print("=" * 60)

# 创建带梯度的张量
x = torch.tensor(3.0, requires_grad=True)
print(f"x = {x}, requires_grad = {x.requires_grad}")

# 前向计算: y = x² + 2x + 1
y = x ** 2 + 2 * x + 1
# dy/dx = 2x + 2 = 2*3 + 2 = 8
print(f"y = x² + 2x + 1 = {y}")

# 反向传播
y.backward()
print(f"y.backward() 后，x.grad = {x.grad}")  # 应为 8
print()

print("=" * 60)
print("2.2 多元函数的梯度")
print("=" * 60)

# y = x1² * x2 + x3
x1 = torch.tensor(2.0, requires_grad=True)
x2 = torch.tensor(3.0, requires_grad=True)
x3 = torch.tensor(4.0, requires_grad=True)

y = x1**2 * x2 + x3
print(f"y = x1² * x2 + x3 = {y.item():.1f}")

y.backward()

# ∂y/∂x1 = 2*x1*x2 = 2*2*3 = 12
# ∂y/∂x2 = x1² = 4
# ∂y/∂x3 = 1
print(f"∂y/∂x1 = {x1.grad}")  # 12
print(f"∂y/∂x2 = {x2.grad}")  # 4
print(f"∂y/∂x3 = {x3.grad}")  # 1
print()

print("=" * 60)
print("2.3 向量/矩阵的梯度（雅可比）")
print("=" * 60)

# 标量对向量的梯度
x = torch.randn(3, requires_grad=True)
y = x.sum()  # y 是标量
y.backward()
print(f"x = {x.data}")
print(f"y = x.sum(), ∂y/∂x = {x.grad}  # 全是 1")
print()

# 向量的情况需要传入 gradient 参数
x = torch.randn(3, requires_grad=True)
v = torch.randn(3)
y = x * 2  # y 是向量

# backward() 只接受标量输出。向量需要传入外部梯度
# y.backward()  # 会报错！
# 正确做法：传入与 y 同形的 gradient 张量
y.backward(v)  # 等价于先计算 v·y 再反向传播
print(f"x = {x.data}")
print(f"v = {v}")
print(f"y = x * 2")
print(f"x.grad = {x.grad}  # = 2 * v")
print()

print("=" * 60)
print("2.4 梯度累加与清零")
print("=" * 60)

x = torch.tensor(1.0, requires_grad=True)

for i in range(3):
    y = x ** 2
    y.backward()
    print(f"第 {i+1} 次 backward() 后, x.grad = {x.grad}")

print("\n⚠️ 梯度会累加！需要手动清零：")

x = torch.tensor(1.0, requires_grad=True)
for i in range(3):
    y = x ** 2
    y.backward()
    print(f"第 {i+1} 次 backward() 后, x.grad = {x.grad}")
    x.grad.zero_()  # 清零梯度

print("\n✅ 使用 zero_() 清零后，每次的梯度都是独立的")
print()

print("=" * 60)
print("2.5 detach() - 从计算图中分离")
print("=" * 60)

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2          # y 在计算图中
z = y.detach()      # z 脱离计算图

print(f"requires_grad: x={x.requires_grad}, y={y.requires_grad}, z={z.requires_grad}")
print(f"grad_fn: y={y.grad_fn}, z={z.grad_fn}")
# grad_fn 为 None 表示该张量是叶子节点或已脱离计算图

# 尝试对 z 反向传播
# z.backward()  # 会报错，z 不在计算图中
print("z.detach() 后无法 backward()，因为没有 grad_fn")
print()

print("=" * 60)
print("2.6 torch.no_grad() - 禁用梯度计算")
print("=" * 60)

x = torch.tensor(2.0, requires_grad=True)

# 正常计算
with torch.no_grad():
    y = x ** 2 + 3 * x + 1
    print(f"no_grad 内: y={y}, requires_grad={y.requires_grad}")

# 等效写法
@torch.no_grad()
def compute(x):
    return x ** 2 + 3 * x + 1

z = compute(x)
print(f"@no_grad 装饰的函数: z={z}, requires_grad={z.requires_grad}")
print("使用场景：模型推理时不需要梯度，可节省显存")

# 手动设置 requires_grad
x.requires_grad_(False)  # 原地修改
print(f"\nrequires_grad_(False) 后: x.requires_grad={x.requires_grad}")
print()

print("=" * 60)
print("2.7 计算图可视化（文本版）")
print("=" * 60)

x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# 线性回归: y = wx + b
u = w * x      # 乘法节点
y = u + b      # 加法节点
loss = (y - 5) ** 2  # 均方误差

print(f"x={x.item()}, w={w.item()}, b={b.item()}")
print(f"y_pred = wx + b = {y.item()}")
print(f"loss = (y_pred - 5)² = {loss.item()}")
print()
print("计算图结构:")
print(f"  x  ─┐")
print(f"  w  ─┤  * (u)  ─┐")
print(f"  b  ─────────┤  + (y)  ─  -  ─  ² (loss)")
print()
print("反向传播 ：loss → y → u → (w, x, b)")

loss.backward()
print(f"\n梯度:")
print(f"  ∂loss/∂w = {w.grad}")  # 2*(y-5)*x = 2*(7-5)*2 = 8
print(f"  ∂loss/∂x = {x.grad}")  # 2*(y-5)*w = 2*(7-5)*3 = 12
print(f"  ∂loss/∂b = {b.grad}")  # 2*(y-5)*1 = 2*(7-5)*1 = 4

# 手动验证
dy = 2 * (y.item() - 5)
dw = dy * x.item()
dx = dy * w.item()
db = dy * 1
print(f"\n手动计算验证:")
print(f"  dloss/dy = 2*(y-5) = {dy}")
print(f"  dloss/dw = dy * x = {dw}")
print(f"  dloss/dx = dy * w = {dx}")
print(f"  dloss/db = dy * 1 = {db}")
print()

print("=" * 60)
print("2.8 常见梯度陷阱")
print("=" * 60)

# 陷阱1: 原地修改 requires_grad 张量
print("陷阱1 - 原地修改带梯度的张量:")
x = torch.tensor([1.0, 2.0], requires_grad=True)
try:
    x += 1  # 原地修改会破坏计算图
except RuntimeError as e:
    print(f"  ❌ x += 1 报错: {str(e)[:60]}...")
print(f"  ✅ 解决方法: x = x + 1 (非原地), 即 y = x + 1")
print()

# 陷阱2: 忘记清零梯度
print("陷阱2 - 忘记清零梯度（已在 2.4 演示）")
print()

# 陷阱3: 标量/非标量
print("陷阱3 - 非标量 backward:")
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2  # y 是向量
# y.backward()  # 会报错
print(f"  ❌ 向量 y = x² 不能直接 backward()")
print(f"  ✅ 需要先转为标量: y.sum().backward() 或传入 gradient 参数")
print()

print("=" * 60)
print("✅ 第 2 步完成！")
print("=" * 60)
