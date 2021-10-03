import torch
from torch import nn
from torch.nn.functional import pad


class IRUNet(nn.Module):

    def __init__(self, train=True):
        super(IRUNet, self).__init__()

        self.encode1_conv = ConvBlock(3, 64)
        self.encode2_conv = ConvBlock(64, 128)
        self.encode3_conv = ConvBlock(128, 256)
        self.encode4_conv = ConvBlock(256, 512)
        self.encode5_conv = ConvBlock(512, 1024)

        self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.attention_block = AttentionBlock(1024)

        self.decode4 = DecodeBlock(1024, 512)
        self.decode3 = DecodeBlock(512, 256)
        self.decode2 = DecodeBlock(256, 128)
        self.decode1 = DecodeBlock(128, 64)

        self.output_block = nn.Conv2d(64, 1, kernel_size=1)

        self.train = train
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(f"x:{x.shape}")
        c1 = self.encode1_conv(x)
        # print(f"c1:{c1.shape}")
        p1 = self.pooling(c1)
        # print(f"p1:{p1.shape}")
        c2 = self.encode2_conv(p1)
        # print(f"c2:{c2.shape}")
        p2 = self.pooling(c2)
        # print(f"p2:{p2.shape}")
        c3 = self.encode3_conv(p2)
        # print(f"c3:{c3.shape}")
        p3 = self.pooling(c3)
        # print(f"p3:{p3.shape}")
        c4 = self.encode4_conv(p3)
        # print(f"c4:{c4.shape}")
        p4 = self.pooling(c4)
        # print(f"p4:{p4.shape}")
        c5 = self.encode5_conv(p4)
        # print(f"c5:{c5.shape}")
        a1 = self.attention_block(c5)
        # print(f"a1:{a1.shape}")

        d4 = self.decode4(a1, c4)
        # print(f"d4:{d4.shape}")
        d3 = self.decode3(d4, c3)
        # print(f"d3:{d3.shape}")
        d2 = self.decode2(d3, c2)
        # print(f"d2:{d2.shape}")
        d1 = self.decode1(d2, c1)
        # print(f"d1:{d1.shape}")

        out = self.output_block(d1)
        if self.train:
            return out, self.sigmoid(out)
        else:
            return self.sigmoid(out)


class AttentionBlock(nn.Module):

    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()

        self.conv_d1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.conv_d2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, dilation=6, padding=1)
        self.conv_d3 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, dilation=12, padding=1)
        self.conv_d4 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, dilation=18, padding=1)

        self.conv1 = nn.Conv2d(in_channels=4 * in_channels, out_channels=in_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.conv2 = nn.Conv2d(in_channels=2 * in_channels, out_channels=in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        d1 = self.conv_d1(x)
        d2 = self.conv_d1(x)
        d3 = self.conv_d1(x)
        d4 = self.conv_d1(x)

        d_cat = torch.cat([d1, d2, d3, d4], dim=1)

        o1 = self.sigmoid(self.conv1(d_cat))

        o2 = torch.cat([torch.multiply(x, o1), x], dim=1)
        o3 = self.conv2(o2)
        return o3


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.LeakyReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.act3 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        c1 = self.conv1(x)
        o1 = self.act1(self.bn1(c1))

        c2 = self.conv2(o1)
        o2 = self.act2(self.bn2(c1 + c2))

        c3 = self.conv3(o2)
        o3 = self.act3(self.bn3(c2 + c3))

        return o3


class DecodeBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DecodeBlock, self).__init__()

        self.up_sample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x_decode, x_encode):
        x_up_sampled = self.up_sample(x_decode)
        # print(f"upsampled to: {x_up_sampled.shape}")
        delta_y = x_encode.shape[2] - x_up_sampled.shape[2]
        delta_x = x_encode.shape[3] - x_up_sampled.shape[3]

        x_padded = pad(x_up_sampled, [delta_x // 2, delta_x - delta_x // 2,
                                      delta_y // 2, delta_y - delta_y // 2])

        return self.conv(torch.cat([x_padded, x_encode], dim=1))
        # return self.conv(torch.cat([x_up_sampled, x_encode], dim=1))


if __name__ == '__main__':
    x = torch.rand((2, 3, 240, 320))
    model = IRUNet()
    o = model(x)
    print(o[0].shape)
