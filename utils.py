import glob
# 导入 glob 模块：glob 可以用通配符（如 "*"）查找符合规则的文件路径
import os
# 导入 os 模块：os 提供路径拼接、目录判断等操作系统相关功能

# 导入 numpy，并给它起一个简短的别名 np，方便后面书写
import numpy as np
# 导入 PyTorch 主模块，后面用它把 numpy 数组转成张量
import torch
# 从 PyTorch 的数据工具模块中导入 Dataset 基类；自定义数据集必须继承它
from torch.utils.data import Dataset


# 定义类 DenoisingDataset，括号里的 Dataset 表示继承 PyTorch 的 Dataset 基类
class DenoisingDataset(Dataset):
    # 下面的三引号字符串是类的说明文档（docstring），解释这个类的作用
    """RGB 去噪任务数据集，输入为退化图，目标为干净图。"""

    # __init__ 是 Python 的初始化方法：创建对象时自动执行；data_dir 是数据根目录参数
    def __init__(self, data_dir):
        # super() 获取父类，调用父类的 __init__ 完成基类自身的初始化
        super().__init__()
        # 把参数 data_dir 保存为对象属性 self.data_dir，之后整个类都能使用
        self.data_dir = data_dir
        # 定义一个属性 sample_dirs，保存所有样本子目录的列表，并用 sorted() 排序
        self.sample_dirs = sorted(
            # 这是列表推导式：遍历 glob.glob() 找到的路径，只保留 os.path.isdir() 判断为目录的项
            [d for d in glob.glob(os.path.join(data_dir, "*")) if os.path.isdir(d)]
        )

    # __len__ 是 Python 的特殊方法：让 len(dataset) 能返回样本总数
    def __len__(self):
        # 样本总数等于样本目录列表的长度
        return len(self.sample_dirs)

    # __getitem__ 是 Python 的特殊方法：让 dataset[idx] 能按下标取出一条样本
    def __getitem__(self, idx):
        # 根据下标 idx 从列表中得到对应的样本目录路径
        sample_path = self.sample_dirs[idx]
        # 用 os.path.join 拼接出当前样本里的退化图文件路径（degraded.npy）
        degraded_path = os.path.join(sample_path, "degraded.npy")
        # 拼接出干净图文件路径（clean.npy）
        clean_path = os.path.join(sample_path, "clean.npy")
        # 拼接出掩码文件路径（mask.npy）
        mask_path = os.path.join(sample_path, "mask.npy")

        # np.load 读取 .npy 文件；.astype(np.float32) 把数组转成 32 位浮点数类型
        degraded_data = np.load(degraded_path).astype(np.float32)
        # 读取干净图数组，并同样转为 float32
        clean_data = np.load(clean_path).astype(np.float32)
        # 读取掩码数组，并转为 float32
        mask_data = np.load(mask_path).astype(np.float32)

        # torch.from_numpy 把 numpy 数组转成 PyTorch 张量；permute(2, 0, 1) 把 HWC 维度顺序改成 CHW；contiguous() 让内存排列连续
        degraded_tensor = torch.from_numpy(degraded_data).permute(2, 0, 1).contiguous()
        # 干净图同样从 HWC 改成 CHW
        clean_tensor = torch.from_numpy(clean_data).permute(2, 0, 1).contiguous()
        # 掩码原本是二维 HxW；unsqueeze(0) 在维度 0 处增加一维，变成 1xHxW，便于和图像拼接
        mask_tensor = torch.from_numpy(mask_data).unsqueeze(0)

        # 返回三个张量：退化图、干净图、掩码
        return degraded_tensor, clean_tensor, mask_tensor
