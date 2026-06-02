"""
PyTorch 学习项目 - 工具模块
提供各步骤共用的辅助函数
"""

import torch
import random
import numpy as np


def set_seed(seed: int = 42):
    """固定随机种子，保证实验结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """自动检测可用设备"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def print_tensor_info(tensor: torch.Tensor, name: str = "Tensor"):
    """打印张量基本信息"""
    print(f"--- {name} ---")
    print(f"  shape: {tensor.shape}")
    print(f"  dtype: {tensor.dtype}")
    print(f"  device: {tensor.device}")
    print(f"  requires_grad: {tensor.requires_grad}")
    if tensor.numel() < 20:
        print(f"  data:\n{tensor.data}")
    print()
