import torch

from utils import set_seed, get_device
import numpy as np

# tensor = 张量
# 维度       名称        样子         距离
# 0维    标量scalar    单个数字     torch.tensor(5) -> 5
# 1维    向量vector    一串数字     torch.tensor([1,2,3,4]) -> [1,2,3,4]
# 2维    矩阵matrix    行列表格     [[1,2],[3,4]]
# 3维    高阶张量       堆叠矩阵     图片[C,H,W] 通道、高、宽


# 固定随机种子，让代码每次运行结果完全一样
set_seed(42)

# 张量
a = torch.tensor([1, 2, 3])
b = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

print('一维张量a:', a, ',\na.size():', a.size(), ',a.shape:', a.shape)
print('二维张量b:', b, ',\nb.size():', b.size(), ',b.shape:', b.shape)

# 工厂方法
zeros = torch.zeros(3, 4)  # 全是0
ones = torch.ones(2, 3)  # 全是1
rand = torch.rand(2, 3)  # [0,1)随机数
randn = torch.randn(2, 3)  # 标准正态分布 N(0,1)
arange = torch.arange(0, 10, 1)  # 等差数列      从0-10 不包含10，每次+1
linspace = torch.linspace(0, 1, 2)  # 等间距    从0-1 均分2个点
linspace02 = torch.linspace(0, 1, 5)  # 等间距    从0-1 均分5个点

print('\n\n\n\n常用工厂方法:\n')
print(f'zeros(3,4)  shape = {zeros.shape} , \nzeros = \n{zeros}')
print(f'ones(2,3)   shape = {ones.shape}  , \nones = \n{ones}')
print(f'rand(2,3)   shape = {rand.shape}  , \nrand = \n{rand}')
print(f'randn(2,3)  shape = {randn.shape}  , \nrandn = \n{randn}')
print(f'arange(0, 10, 1)   shape = {arange.shape}  , \narange = \n{arange}')
print(f'linspace(0, 1, 2)   shape = {linspace.shape}  , \nlinspace = \n{linspace}')
print(f'linspace02(0, 1, 2)   shape = {linspace02.shape}  , \nlinspace02 = \n{linspace02}')

print('\n\n\n\n')

# 从numpy数组创建
np_arr = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
from_np = torch.from_numpy(np_arr)  # 共享内存  修改则原始也变
from_np_copy = torch.tensor(np_arr)  # 拷贝
from_np_copy2 = torch.tensor(np_arr)
print('从 Numpy 创建:')
print(f'from_numpy(共享内存): \n{from_np}')
print(f'from_numpy_copy(拷贝): \n{from_np_copy}')
print(f'from_np_copy2(拷贝): \n{from_np_copy2}')

print('\n\n\n\n')

# 创建与已有张量相同属性的张量
c = torch.randn(2, 3)
like_zeros = torch.zeros_like(c)
like_ones = torch.ones_like(c)
print(f'c  保持 shape={c.shape},dtype = {c.dtype} , \n{c}')
print(f'zeros_like  保持 shape={like_zeros.shape},dtype = {like_zeros.dtype} , \n{like_zeros}')
print(f'ones_like   保持 shape={like_ones.shape},dtype = {like_ones.dtype} , \n{like_ones}')

print('\n\n\n\n')
# 张量属性
x = torch.randn(2, 3, 4)  # 2个块  3行 4列
print(f'shape: {x.shape}  #形状')
print(f'size: {x.size()}  #形状')
print(f'ndim: {x.ndim}  #维度数(秩)')  # 3
print(f'dtype: {x.dtype}  #数据类型')  # torch.float32
print(f'device: {x.device}  #所在设备')
print(f'numel: {x.numel()}  #元素总数')  # 2*3*4 = 24
print(f'stride: {x.stride()}  #步长(各维度的内存跨度)')  # (12,4,1)    跳入下一个维度的步数(3*4,4,1)

print('\n\n\n\n')
# 切片、索引
print(torch.arange(12))
x = torch.arange(12).reshape(3, 4)
print(f'原始张量 (3*4) : \n{x}\n')
print(f'x[0, 0] = \n{x[0, 0]}  # 单个元素->标量')
print(f'x[0] = \n{x[0]}  # 第0行')
print(f'x[1] = \n{x[1]}  # 第1行')
print(f'x[:, 1] = \n{x[:, 1]}  # 第1列')
print(f'x[:, 2] = \n{x[:, 2]}  # 第2列')
print(f'x[:2, 1] = \n{x[:2, 1:]}  # 前2行 + 后3列')
print(f'x[:2, -3] = \n{x[:2, -3:]}  # 前2行 + 后3列')
print(f'x[::2]  = \n{x[::2]}    # 每隔一行')
print(f'x[x>5] = {x[x > 5]}       # 布尔索引')

print('\n\n\n\n')
# 修改切片会影响原张量
y = x[:2, :2]  # 前两行两列
print(y)
y[0, 0] = 999
print(f'\n修改切片 y(y[0, 0]=999)后 ，x 也被修改了:\n\n{x}')

# 形状操作
x = torch.arange(12)
print(f'原始x:\n{x}')
# reshape -1 表示自动推导
print(f'reshape(3, 4):\n {x.reshape(3, 4)}')
print(f'reshape(2, -1):\n {x.reshape(2, -1)}')
# view: 与 reshape 类似，但要求内存连续
print(f'view(3, 4):\n{x.view(3, 4)}')

# 添加/移除维度
print(f'unsqueeze(0).shape: {x.unsqueeze(0).shape}  # 在0维前加1维')
print(f'unsqueeze(1).shape: {x.unsqueeze(1).shape}  # 在1维前加1维')
print(torch.tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]]))
print(torch.tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]]).shape)
print(
    f'squeeze().shape: {torch.tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]]).unsqueeze(0).squeeze(0).shape}  # 移除大小为 1 的维度')
print(
    f'squeeze().shape: {torch.tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]]).unsqueeze(0).squeeze(0)}  # 移除大小为 1 的维度')

# 展平 flatten
print(f'flatten: {x.reshape(2, 2, 3).flatten()}')

print('\n\n\n\n')
print('拼接与堆叠')

a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])
print(f'a = \n{a}')
print(f'b = \n{b}')

# cat 延已有维度拼接
print(f'\ntorch.cat([a,b],dim = 0:\n{torch.cat([a, b], dim=0)} # 纵向拼接)')
print(f'\ntorch.cat([a,b],dim = 1:\n{torch.cat([a, b], dim=1)} # 横向拼接)')

# stack: 新增一个维度
print(f'\ntorch.stack([a,b],dim=0):\n{torch.stack([a, b], dim=0)} # shape=(2,2,2)')
print(f'torch.stack([a,b],dim=1):\n{torch.stack([a, b], dim=1)} # shape=(2,2,2)')

print('\n\n\n\n')
# 张量运算
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(f' a = {a} \n b = {b}')
print(f'a+b = {a + b} # 逐元素加法')
print(f'a-b = {a - b} # 逐元素减法')
print(f'a*b = {a * b} # 逐元素乘法(非矩阵乘)')
print(f'a/b = {a / b} # 逐元素除法')
print(f'a@b = {a @ b} # 点积(1D 向量内积)')  # a * b 然后所有元素 和
print(f'torch.dot(a,b) = {torch.dot(a, b)} # 向量点积')
print(f'a ** 2 = {a ** 2} # 逐元素幂')
print(f'a.sum() = {a.sum()} # 求和')
print(f'a.mean() = {a.mean()} # 均值')
print(f'a.max() = {a.max()} # 最大值')
print(f'a.argmax() = {a.argmax()} # 最大值索引')

# 广播机制
print('\n\n\n\n广播机制')
x = torch.ones(3, 1)
y = torch.tensor([1.0, 2.0, 3.0, 4.0])
print(f'x.shape = {x.shape} \nx={x}')
print(f'y.shape = {y.shape} \ny={y}')
print(f'(x+y).shape = {(x + y).shape} \nx + y ={x + y}')

# 标量 0维张量
print('\n\n\n\n标量 0维张量')
scalar = torch.tensor(3.14)
print(f'标量:{scalar}')
print(f'shape:{scalar.shape} # 空列表，表示 0 维')
print(f'item():{scalar.item()} # 提取 Python 数值')

# 许多统计操作返回标量
print('\n\n\n\n许多统计操作返回标量')
loss = torch.randn(100).sum()
print(f'loss = {loss}')
print(f'loss.shape = {loss.shape}')
print(f'loss.item() = {loss.item()}')

# 数据类型转换
print('\n\n\n\n数据类型转换')
x = torch.tensor([1, 2, 3])
print(f'int64: {x} , dtype={x.dtype}')

x_float = x.float()     # -> float32
x_double = x.double()   # -> float64
x_half = x.half()   # -> float16
x_long = x_float.long()   # -> int64

print(f'float():{x_float}, dtype={x_float.dtype}')
print(f'double():{x_double}, dtype={x_double.dtype}')
print(f'half():{x_half}, dtype={x_half.dtype}')
print(f'long():{x_long}, dtype={x_long.dtype}')

# type() 和 to() 方法
print('\n\n\n\n')
print(f'\nto(dtype=float32):{x.to(dtype=torch.float32)}')

print('\n\n\n\n设备转移')
device = get_device()
print(f'当前设备:{device}')
x = torch.randn(3,3)
print(f'创建在 CPU: device={x.device}')

if torch.cuda.is_available():
    x_gpu = x.to('cuda')    # 或 x.cuda()
    print(f'转移到 GPU: device={x_gpu.device}')
    x_cpu = x_gpu.to('cpu')     # 或 x_gpu.cpu()
    print(f'转移回 CPU: device={x_cpu.device}')
else:
    print('本机无GPU ， 跳过GPU 转移演示')