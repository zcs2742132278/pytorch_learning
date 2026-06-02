"""
======================================================================
PyTorch 15 步学习导航
======================================================================
按顺序运行每个 step 文件，理解 PyTorch 的核心概念

运行方式:
  python step01_tensor_basics.py
  python step02_autograd.py
  ...
  python step15_tensorboard.py

或者在这里按数字选择运行。
======================================================================
"""

import sys
import subprocess
from pathlib import Path

STEPS = [
    ("01", "step01_tensor_basics.py", "张量 - 标量"),
    ("02", "step02_autograd.py", "自动微分"),
    ("03", "step03_linear_regression.py", "线性回归（原始 + 封装）"),
    ("04", "step04_classification.py", "实现分类"),
    ("05", "step05_model_subclass.py", "模型子类写法"),
    ("06", "step06_dataset.py", "Dataset 重构"),
    ("07", "step07_dataloader.py", "DataLoader 重构"),
    ("08", "step08_data_augmentation.py", "数据增强"),
    ("09", "step09_code_validation.py", "代码校验 + 封装"),
    ("10", "step10_cnn_mnist.py", "CNN 手写数字识别"),
    ("11", "step11_lr_scheduler.py", "学习率衰减"),
    ("12", "step12_model_save.py", "模型参数保存"),
    ("13", "step13_weather_classification.py", "天气分类"),
    ("14", "step14_transfer_learning.py", "迁移学习"),
    ("15", "step15_tensorboard.py", "TensorBoard 使用"),
]


def main():
    print("=" * 60)
    print("  PyTorch 15 步学习路线")
    print("=" * 60)
    print()
    for num, filename, desc in STEPS:
        print(f"  [{num}] {desc}")

    print()
    print("  输入编号运行对应的 step (如: 01)")
    print("  输入 'all' 按顺序运行所有步骤")
    print("  输入 'q' 退出")
    print()

    while True:
        choice = input(">>> ").strip()

        if choice.lower() == 'q':
            print("再见！")
            break
        elif choice.lower() == 'all':
            for num, filename, desc in STEPS:
                print(f"\n{'='*60}")
                print(f"  [{num}] {desc}")
                print(f"{'='*60}")
                filepath = Path(__file__).parent / filename
                subprocess.run([sys.executable, str(filepath)])
        else:
            # 查找对应的 step
            found = False
            for num, filename, desc in STEPS:
                if choice == num:
                    filepath = Path(__file__).parent / filename
                    print(f"\n运行 [{num}] {desc}...")
                    subprocess.run([sys.executable, str(filepath)])
                    found = True
                    break
            if not found:
                print(f"无效输入: {choice}，请输入 01-15, all, 或 q")


if __name__ == "__main__":
    main()
