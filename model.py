import torch
# 导入 PyTorch 主模块，里面包含张量、自动求导等基础功能
import torch.nn as nn
# 导入 PyTorch 的神经网络子模块，并简写成 nn，用于定义网络层


# 定义类 Conv2d，继承 nn.Module（PyTorch 所有网络的基类）
class Conv2d(nn.Module):
    # 初始化方法：创建 Conv2d 对象时自动执行；in_channels 是输入通道数，out_channels 是输出通道数
    def __init__(self, in_channels, out_channels):
        # 调用父类 nn.Module 的初始化方法，完成 PyTorch 内部状态注册
        super().__init__()
        # 创建一个 1x1 卷积层：kernel_size=1 表示卷积核是 1x1，padding=0 表示不做填充
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    # forward 是网络层的前向传播方法：输入张量 x，输出计算结果
    def forward(self, x):
        # 把输入 x 送入之前创建的卷积层 self.conv2d，并把结果返回
        return self.conv2d(x)


# 定义类 ConvBlock：一个由卷积、ReLU、批归一化组成的小模块
class ConvBlock(nn.Module):
    # 初始化方法；kernel_size 默认是 3，stride 默认是 1
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        # 调用父类初始化方法
        super().__init__()
        # isinstance() 判断 kernel_size 是否是一个元组（例如 (1, 7) 表示两个方向的卷积核尺寸）
        if isinstance(kernel_size, tuple):
            # 生成器表达式：对元组里的每个尺寸 size 计算 size // 2，再转成元组作为填充大小
            padding = tuple(size // 2 for size in kernel_size)
        # 如果不是元组，就执行 else 分支
        else:
            # 单个数字时，填充大小等于 kernel_size // 2（// 是整除，向下取整）
            padding = kernel_size // 2
        # nn.Sequential 是一个容器：按顺序依次执行里面的每个层
        self.block = nn.Sequential(
            # 创建二维卷积层，下面是它的参数
            nn.Conv2d(
                # 指定输入通道数
                in_channels,
                # 指定输出通道数
                out_channels,
                # 指定卷积核大小（可以是数字或元组）
                kernel_size=kernel_size,
                # 指定卷积步长
                stride=stride,
                # 指定填充大小，用来保持输出尺寸
                padding=padding,
            # 这一行结束 nn.Conv2d 的参数并闭合调用
            ),
            # ReLU 是激活函数，把负数变成 0；inplace=True 表示直接修改原张量，节省内存
            nn.ReLU(inplace=True),
            # BatchNorm2d 对 out_channels 个通道做批归一化，让训练更稳定
            nn.BatchNorm2d(out_channels),
        # 这一行结束 nn.Sequential 的调用，结果保存进 self.block
        )

    # 前向传播：输入 x
    def forward(self, x):
        # 把 x 依次通过 self.block 里的所有层，并返回结果
        return self.block(x)


# 定义类 MultiConvLayer：并行使用多个卷积分支，再融合结果
class MultiConvLayer(nn.Module):
    # 三引号字符串是这个类的说明文档
    """Parallel local and directional convolution block."""

    # 初始化方法：in_channels 输入通道数，out_channels 输出通道数，stride 默认 1
    def __init__(self, in_channels, out_channels, stride=1):
        # 调用父类初始化
        super().__init__()
        # 分支 1：用 3x3 卷积核提取局部特征
        self.branch3 = ConvBlock(in_channels, out_channels, kernel_size=3, stride=stride)
        # 分支 2：用 (1, 7) 卷积核提取水平方向的长条特征；下面一行是它的参数结束符
        self.branch_h = ConvBlock(
            # 把输入通道数传给 ConvBlock
            in_channels, out_channels, kernel_size=(1, 7), stride=stride
        # 结束 self.branch_h 的调用
        )
        # 分支 3：用 (7, 1) 卷积核提取垂直方向的长条特征
        self.branch_v = ConvBlock(
            # 传入输入通道数、输出通道数和 (7, 1) 卷积核
            in_channels, out_channels, kernel_size=(7, 1), stride=stride
        # 结束 self.branch_v 的调用
        )
        # 融合层：三个分支输出通道数是 out_channels * 3，通过 1x1 卷积压缩回 out_channels
        self.fuse = Conv2d(out_channels * 3, out_channels)

    # 前向传播：输入 x
    def forward(self, x):
        # 把 x 送入 3x3 分支，得到特征 x3
        x3 = self.branch3(x)
        # 把 x 送入水平卷积分支，得到特征 xh
        xh = self.branch_h(x)
        # 把 x 送入垂直卷积分支，得到特征 xv
        xv = self.branch_v(x)
        # torch.cat 把三个特征沿 dim=1（通道维）拼成一个张量
        fusion = torch.cat([x3, xh, xv], dim=1)
        # 用 1x1 卷积融合并返回结果
        return self.fuse(fusion)


# 定义类 UpBlock：负责上采样并和跳跃连接特征合并
class UpBlock(nn.Module):
    # 初始化方法：in_channels 输入通道，skip_channels 跳跃连接通道，out_channels 输出通道
    def __init__(self, in_channels, skip_channels, out_channels):
        # 调用父类初始化
        super().__init__()
        # 上采样模块：由转置卷积、ReLU、批归一化组成
        self.up = nn.Sequential(
            # 创建转置卷积（反卷积），可以把特征图尺寸放大一倍
            nn.ConvTranspose2d(
                # 输入通道数
                in_channels,
                # 输出通道数
                out_channels,
                # 卷积核大小
                kernel_size=3,
                # 步长为 2，配合 output_padding 让尺寸翻倍
                stride=2,
                # 填充大小
                padding=1,
                # 输出额外填充，保证上采样后尺寸正确
                output_padding=1,
            # 结束 nn.ConvTranspose2d 调用
            ),
            # 上采样后经过 ReLU 激活
            nn.ReLU(inplace=True),
            # 再经过批归一化
            nn.BatchNorm2d(out_channels),
        # 结束 nn.Sequential 调用
        )
        # merge 用 1x1 卷积把上采样结果和跳跃连接拼接后的通道数压回 out_channels
        self.merge = Conv2d(out_channels + skip_channels, out_channels)
        # refine 再用多分支卷积进一步细化特征
        self.refine = MultiConvLayer(out_channels, out_channels, stride=1)

    # 前向传播：x 是主路径特征，skip 是编码器传来的跳跃连接特征
    def forward(self, x, skip):
        # 先对 x 做上采样
        x = self.up(x)
        # 把上采样后的 x 和 skip 沿通道维拼接
        x = torch.cat([x, skip], dim=1)
        # 用 1x1 卷积合并通道
        x = self.merge(x)
        # 再经过细化模块后返回
        return self.refine(x)


# 定义类 Encoder：编码器，负责逐级提取特征并缩小空间尺寸
class Encoder(nn.Module):
    # 初始化方法：in_channels 默认是 3（RGB 三通道）
    def __init__(self, in_channels=3):
        # 调用父类初始化
        super().__init__()
        # stem 是入口模块：先用 1x1 卷积升到 16 通道，再用多分支卷积升到 32 通道
        self.stem = nn.Sequential(
            # 输入 in_channels 通道，输出 16 通道
            Conv2d(in_channels, 16),
            # 多分支卷积：16 通道转 32 通道，stride=1 保持尺寸
            MultiConvLayer(16, 32, stride=1),
        # 结束 stem 的 nn.Sequential 调用
        )
        # down1 是第一级下采样：把通道从 32 升到 64，并把空间尺寸缩小一半
        self.down1 = nn.Sequential(
            # 先用 1x1 卷积把 32 通道变成 64 通道
            Conv2d(32, 64),
            # 多分支卷积中 stride=2，让特征图尺寸减半
            MultiConvLayer(64, 64, stride=2),
        # 结束 down1 的 nn.Sequential 调用
        )
        # down2 是第二级下采样：64 通道变 128 通道，尺寸再减半
        self.down2 = nn.Sequential(
            # 1x1 卷积把 64 通道变成 128 通道
            Conv2d(64, 128),
            # stride=2 的多分支卷积继续下采样
            MultiConvLayer(128, 128, stride=2),
        # 结束 down2 的调用
        )
        # down3 是第三级下采样：128 通道变 256 通道，尺寸再次减半
        self.down3 = nn.Sequential(
            # 1x1 卷积把 128 通道变成 256 通道
            Conv2d(128, 256),
            # stride=2 的多分支卷积继续下采样
            MultiConvLayer(256, 256, stride=2),
        # 结束 down3 的调用
        )

    # 前向传播：输入 x
    def forward(self, x):
        # 通过 stem 得到 32 通道特征 s0
        s0 = self.stem(x)
        # 通过 down1 得到 64 通道特征 s1
        s1 = self.down1(s0)
        # 通过 down2 得到 128 通道特征 s2
        s2 = self.down2(s1)
        # 通过 down3 得到 256 通道最深特征 out
        out = self.down3(s2)
        # 返回最深特征以及三个跳跃连接特征 s0、s1、s2
        return out, s0, s1, s2


# 定义类 Decoder：解码器，负责逐步上采样并恢复图像
class Decoder(nn.Module):
    # 初始化方法：out_channels 默认是 3，即最终输出 RGB 三通道
    def __init__(self, out_channels=3):
        # 调用父类初始化
        super().__init__()
        # up1 把 256 通道上采样到 128 通道，并和 s2（128 通道）合并
        self.up1 = UpBlock(256, 128, 128)
        # up2 把 128 通道上采样到 64 通道，并和 s1（64 通道）合并
        self.up2 = UpBlock(128, 64, 64)
        # up3 把 64 通道上采样到 32 通道，并和 s0（32 通道）合并
        self.up3 = UpBlock(64, 32, 32)
        # head 是输出头：进一步处理并把通道压到 3，得到最终去噪图
        self.head = nn.Sequential(
            # 先用多分支卷积细化 32 通道特征
            MultiConvLayer(32, 32, stride=1),
            # 1x1 卷积把 32 通道降到 16 通道
            Conv2d(32, 16),
            # ReLU 激活
            nn.ReLU(inplace=True),
            # 最后一个 1x1 卷积输出 out_channels（默认 3）通道
            Conv2d(16, out_channels),
        # 结束 head 的 nn.Sequential 调用
        )

    # 前向传播：x 是编码器最深特征，s0、s1、s2 是三个跳跃连接
    def forward(self, x, s0, s1, s2):
        # 第一级上采样，并和 s2 合并
        x = self.up1(x, s2)
        # 第二级上采样，并和 s1 合并
        x = self.up2(x, s1)
        # 第三级上采样，并和 s0 合并
        x = self.up3(x, s0)
        # 通过输出头生成最终图像
        return self.head(x)


# 定义类 RGBDenoiseNet：把编码器和解码器组装成完整的去噪网络
class RGBDenoiseNet(nn.Module):
    # 初始化方法：输入默认 4 通道（RGB 三通道加 1 个掩码通道），输出默认 3 通道
    def __init__(self, in_channels=4, out_channels=3):
        # 调用父类初始化
        super().__init__()
        # 创建编码器对象并保存为 self.encoder
        self.encoder = Encoder(in_channels=in_channels)
        # 创建解码器对象并保存为 self.decoder
        self.decoder = Decoder(out_channels=out_channels)

    # 前向传播：xin 是 4 通道输入（退化图 + 掩码）
    def forward(self, xin):
        # 先经过编码器，拿到最深特征 x 和三个跳跃特征 s0、s1、s2
        x, s0, s1, s2 = self.encoder(xin)
        # 再把特征交给解码器，得到输出图像
        output = self.decoder(x, s0, s1, s2)
        # torch.clamp 把输出限制在 [0.0, 1.0] 之间，保证像素值合法
        return torch.clamp(output, 0.0, 1.0)
