import torch
import torch.nn as nn


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        return self.conv2d(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        if isinstance(kernel_size, tuple):
            padding = tuple(size // 2 for size in kernel_size)
        else:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.block(x)


class MultiConvLayer(nn.Module):
    """Parallel local and directional convolution block."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.branch3 = ConvBlock(in_channels, out_channels, kernel_size=3, stride=stride)
        self.branch_h = ConvBlock(
            in_channels, out_channels, kernel_size=(1, 7), stride=stride
        )
        self.branch_v = ConvBlock(
            in_channels, out_channels, kernel_size=(7, 1), stride=stride
        )
        self.fuse = Conv2d(out_channels * 3, out_channels)

    def forward(self, x):
        x3 = self.branch3(x)
        xh = self.branch_h(x)
        xv = self.branch_v(x)
        fusion = torch.cat([x3, xh, xv], dim=1)
        return self.fuse(fusion)


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )
        self.merge = Conv2d(out_channels + skip_channels, out_channels)
        self.refine = MultiConvLayer(out_channels, out_channels, stride=1)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.merge(x)
        return self.refine(x)


class Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.stem = nn.Sequential(
            Conv2d(in_channels, 16),
            MultiConvLayer(16, 32, stride=1),
        )
        self.down1 = nn.Sequential(
            Conv2d(32, 64),
            MultiConvLayer(64, 64, stride=2),
        )
        self.down2 = nn.Sequential(
            Conv2d(64, 128),
            MultiConvLayer(128, 128, stride=2),
        )
        self.down3 = nn.Sequential(
            Conv2d(128, 256),
            MultiConvLayer(256, 256, stride=2),
        )

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        out = self.down3(s2)
        return out, s0, s1, s2


class Decoder(nn.Module):
    def __init__(self, out_channels=3):
        super().__init__()
        self.up1 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up3 = UpBlock(64, 32, 32)
        self.head = nn.Sequential(
            MultiConvLayer(32, 32, stride=1),
            Conv2d(32, 16),
            nn.ReLU(inplace=True),
            Conv2d(16, out_channels),
        )

    def forward(self, x, s0, s1, s2):
        x = self.up1(x, s2)
        x = self.up2(x, s1)
        x = self.up3(x, s0)
        return self.head(x)


class RGBDenoiseNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels)
        self.decoder = Decoder(out_channels=out_channels)

    def forward(self, xin):
        x, s0, s1, s2 = self.encoder(xin)
        output = self.decoder(x, s0, s1, s2)
        return torch.clamp(output, 0.0, 1.0)
