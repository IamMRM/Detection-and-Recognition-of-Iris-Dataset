from torch import nn
import torch


class SiameseNet(nn.Module):

    def __init__(self):
        super(SiameseNet, self).__init__()

        ## Model 1
        # self.conv_net = nn.Sequential(
        #     ConvBlock(in_channels=3, out_channels=64),
        #     ConvBlock(in_channels=64, out_channels=128),
        #     ConvBlock(in_channels=128, out_channels=256),
        #     ConvBlock(in_channels=256, out_channels=512),
        #     ConvBlock(in_channels=512, out_channels=1024, kernel1=3, kernel2=1),
        #     # ConvBlock(in_channels=1024, out_channels=1024, kernel1=1, kernel2=1),
        #
        #     nn.Dropout2d(p=0.2)
        # )
        # input_fc_shape = 28672
        # self.fc_net = nn.Sequential(
        #     Flatten(),
        #     nn.Linear(in_features=input_fc_shape, out_features=2048),
        #     nn.Dropout(p=0.2),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(in_features=2048, out_features=1024),
        #     nn.Dropout(p=0.2),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(in_features=1024, out_features=64)
        # )

        ## Model 2
        # self.conv_net = nn.Sequential(
        #     ConvBlock(in_channels=3, out_channels=32, kernel1=5),
        #     ConvBlock(in_channels=32, out_channels=64),
        #     ConvBlock(in_channels=64, out_channels=128),
        #     ConvBlock(in_channels=128, out_channels=256),
        #     ConvBlock(in_channels=256, out_channels=256, kernel1=1, kernel2=1),
        #
        #     nn.Dropout2d(p=0.2)
        # )
        # input_fc_shape = 10240
        # self.fc_net = nn.Sequential(
        #     Flatten(),
        #     nn.Linear(in_features=input_fc_shape, out_features=2048),
        #     nn.BatchNorm1d(2048),
        #     nn.Dropout(p=0.2),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(in_features=2048, out_features=1024),
        #     nn.BatchNorm1d(1024),
        #     nn.Dropout(p=0.2),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(in_features=1024, out_features=64)
        # )


        ## Model 3
        # self.conv_net = nn.Sequential(
        #     ConvBlock(in_channels=3, out_channels=32, kernel1=5, kernel2=5),
        #     ConvBlock(in_channels=32, out_channels=64, kernel1=5, kernel2=5),
        #     ConvBlock(in_channels=64, out_channels=128),
        #     ConvBlock(in_channels=128, out_channels=256),
        #     ConvBlock(in_channels=256, out_channels=256, kernel1=1, kernel2=1),
        #
        #     nn.Dropout2d(p=0.2)
        # )
        # input_fc_shape = 8960
        # self.fc_net = nn.Sequential(
        #     Flatten(),
        #     nn.Linear(in_features=input_fc_shape, out_features=2048),
        #     nn.BatchNorm1d(2048),
        #     nn.Dropout(p=0.2),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(in_features=2048, out_features=1024),
        #     nn.BatchNorm1d(1024),
        #     nn.Dropout(p=0.2),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(in_features=1024, out_features=64)
        # )

        ## Model 4
        # self.conv_net = nn.Sequential(
        #     ConvBlock(in_channels=3, out_channels=32, kernel1=5, kernel2=5),
        #     ConvBlock(in_channels=32, out_channels=64, kernel1=5, kernel2=5),
        #     ConvBlock(in_channels=64, out_channels=128),
        #     ConvBlock(in_channels=128, out_channels=256),
        #     ConvBlock(in_channels=256, out_channels=256, kernel1=1, kernel2=1),
        #
        #     nn.Dropout2d(p=0.2)
        # )

        self.conv1 = ConvBlock(in_channels=3, out_channels=32, kernel1=5, kernel2=5)
        self.conv2 = ConvBlock(in_channels=32, out_channels=64, kernel1=5, kernel2=5)
        self.conv3 = ConvBlock(in_channels=64, out_channels=128)
        self.conv4 = ConvBlock(in_channels=128, out_channels=256)
        self.conv5 = ConvBlock(in_channels=256, out_channels=256, kernel1=1, kernel2=1)

        self.conv_drop = nn.Dropout2d(p=0.2)

        input_fc_shape = 8960
        self.fc_net = nn.Sequential(
            Flatten(),
            nn.Linear(in_features=input_fc_shape, out_features=2048),
            nn.BatchNorm1d(2048),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=2048, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU()
        )

        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
        self.fc_net.apply(init_weights)

    def forward_once(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        c = self.conv_drop(c5)
        return self.fc_net(c)

        # return self.fc_net(self.conv_net(x))

    def forward(self, x1, x2=None):
        if x2 is not None:
            return self.forward_once(x1), self.forward_once(x2)
        else:
            return self.forward_once(x1)


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel1=3, kernel2=3, pad1=0, pad2=0):
        super(ConvBlock, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel1, padding=pad1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel2, padding=pad2),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.net(x))


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return torch.flatten(x, start_dim=1)


if __name__ == '__main__':
    import torchsummary
    model = SiameseNet()
    model(torch.rand((2, 3, 240, 320)))
    torchsummary.summary(SiameseNet(), torch.rand((2, 3, 240, 320)))
