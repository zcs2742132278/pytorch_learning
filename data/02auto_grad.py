'''
自动微分

1.理解计算图(Computation Graph) 的构建与反向传播
2.掌握 requires_grad 属性与backward() 方法
3.理解梯度累加与清零机制
4.掌握detach()、no_grad()、zero_grad() 的用法
5.理解叶子节点与非叶子节点的梯度规则
'''
import torch
from utils import set_seed

set_seed(42)

print('自动微分\n')
# 创建带梯度的张量
x = torch.tensor(3.0, requires_grad=True)
# x.grad = 损失函数 loss 对 x 的导数（梯度值） 反向传播前x.grad = None
# 常用求导公式
# 1.(x^n)' = nx^n-1 -> (x²)' = 2x
# 2.(kx)' = k -> (4x)' = 4
# 3.常数倒数 => 0  100'=0
print(f'x = {x}, requires_grad = {x.requires_grad}, x.grad = {x.grad}')     # 反向传播前x.grad = None

# 前向计算:y = x² + 2x + 1
y = x ** 2 + 2 * x + 1
# dy/dx = 2x + 2 = 2*3 +2 = 8
print(f'y = x² + 2x + 1 = {y}')

# 反向传播  反向传播前x.grad = None
y.backward()
# y = x² + 2x + 1  -> x' = x.grad = dy/dx = 2x + 2 = 8
print(f'y.backward() 后，x.grad = {x.grad} ')  # 应为 8

print('\n\n\n\n多元函数的梯度')
# y = x1² * x2 + x3
x1 = torch.tensor(2.0, requires_grad=True)
x2 = torch.tensor(3.0, requires_grad=True)
x3 = torch.tensor(4.0, requires_grad=True)

y = x1**2 * x2 + x3
print(f'y = x1² * x2 + x3 = {y.item():.2f}') #  :.1f 保留一位小数
# 反向传播
y.backward()
# ∂y/∂x1 = 2*x1*x2 = 2*2*3 = 12
# ∂y/∂x2 = x1² = 4
# ∂y/∂x3 = 1
print(f'∂y/∂x1 = {x1.grad}')    # y=x1²*x2+x3 -> x1' = 2 * x1 * 3 = 2*2*3 =12
print(f'∂y/∂x2 = {x2.grad}')    # y=4*x2+4 -> x2' = 4
print(f'∂y/∂x3 = {x3.grad}')    # y=4*3+x3 -> x3' = 1

print('\n\n\n\n向量/矩阵的梯度(雅可比)')
# 标量对向量的梯度
x = torch.randn(3, requires_grad=True)  # 向量
print(x)
y = x.sum() # y是标量
y.backward()
print(f'x = {x.data}')
print(f'y = x.sum(), ∂y/∂x = {x.grad} # 全是 1')

# 向量的情况需要传入 gradient 参数
x = torch.randn(3, requires_grad=True)
v = torch.randn(3)
y = x * 2  # y 是向量

# backward() 只接受标量输出，向量需要传入外部梯度
# y.backward() #会报错！
# 正确做法:传入与 y 同形的 gradient 张量
y.backward(v) # 等价于先计算 v * y 再反向传播
print(f'x = {x.data}')
print(f'v = {v}')
print(f'y = x * 2')
print(f'x.grad = {x.grad}  # = 2 * v = x\' * v')

m = x ** 2 + 2 * x + 1

m.backward(v)
print(x.grad) #(2x + 2) * v