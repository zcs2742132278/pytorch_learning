from utils import set_seed

# tensor = 张量
# 维度       名称        样子         距离
# 0维    标量scalar    单个数字     torch.tensor(5) -> 5
# 1维    向量vector    一串数字     torch.tensor([1,2,3,4]) -> [1,2,3,4]
# 2维    矩阵matrix    行列表格     [[1,2],[3,4]]
# 3维    高阶张量       堆叠矩阵     图片[C,H,W] 通道、高、宽


# 固定随机种子，让代码每次运行结果完全一样
set_seed(42)