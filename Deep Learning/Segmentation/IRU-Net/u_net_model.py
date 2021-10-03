import torch
from torch import nn
from torch.nn.functional import pad


class UNet(nn.Module):

    def __init__(self, train=True):
        super(UNet, self).__init__()

        self.encode1_conv = ConvBlock(3, 64)
        self.encode2_conv = ConvBlock(64, 128)
        self.encode3_conv = ConvBlock(128, 256)
        self.encode4_conv = ConvBlock(256, 512)
        self.encode5_conv = ConvBlock(512, 1024)

        self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)

        self.decode4 = DecodeBlock(1024, 512)
        self.decode3 = DecodeBlock(512, 256)
        self.decode2 = DecodeBlock(256, 128)
        self.decode1 = DecodeBlock(128, 64)

        self.output_block = nn.Conv2d(64, 1, kernel_size=1)

        self.train = train
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        c1 = self.encode1_conv(x)
        p1 = self.pooling(c1)
        c2 = self.encode2_conv(p1)
        p2 = self.pooling(c2)
        c3 = self.encode3_conv(p2)
        p3 = self.pooling(c3)
        c4 = self.encode4_conv(p3)
        p4 = self.pooling(c4)
        c5 = self.encode5_conv(p4)
        # p5 = self.pooling(c5)

        d4 = self.decode4(c5, c4)
        d3 = self.decode3(d4, c3)
        d2 = self.decode2(d3, c2)
        d1 = self.decode1(d2, c1)

        out = self.output_block(d1)
        if self.train:
            return out, self.sigmoid(out)
        else:
            return self.sigmoid(out)


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class DecodeBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DecodeBlock, self).__init__()

        self.up_sample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x_decode, x_encode):
        x_up_sampled = self.up_sample(x_decode)
        delta_y = x_encode.shape[2] - x_up_sampled.shape[2]
        delta_x = x_encode.shape[3] - x_up_sampled.shape[3]

        x_padded = pad(x_up_sampled, [delta_x // 2, delta_x - delta_x // 2,
                                      delta_y // 2, delta_y - delta_y // 2])

        return self.conv(torch.cat([x_padded, x_encode], dim=1))


if __name__ == '__main__':
    x = torch.rand((2, 3, 240, 320))
    model = UNet()
    o = model(x)
