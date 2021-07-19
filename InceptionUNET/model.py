import torch
import torch.nn as nn


class DenseBlock(nn.Module):
    def __init__(self, in_channels):
        super(DenseBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(in_channels, )

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, stride=1, dilation=8, padding=8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, dilation=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, dilation=4, padding=4)
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=8, kernel_size=3, stride=1, dilation=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3, stride=1, dilation=16, padding=16)

    def forward(self, x):
        bn = self.bn(x)
        conv1 = self.relu(self.conv1(bn))
        conv2 = self.relu(self.conv2(conv1))
        # Concatenate in channel dimension
        c2_dense = self.relu(torch.cat([conv1, conv2], 1))

        conv3 = self.relu(self.conv3(c2_dense))
        c3_dense = self.relu(torch.cat([conv1, conv2, conv3], 1))

        conv4 = self.relu(self.conv4(c3_dense))
        c4_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4], 1))

        conv5 = self.relu(self.conv5(c4_dense))
        c5_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5], 1))

        return c5_dense


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_factor=2):
        super(InceptionBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(in_channels, )

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8 * out_factor, kernel_size=3, dilation=1,
                               stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=8 * out_factor, kernel_size=3, dilation=2,
                               stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=8 * out_factor, kernel_size=3, dilation=4,
                               stride=1, padding=4)
        self.conv4 = nn.Conv2d(in_channels=in_channels, out_channels=8 * out_factor, kernel_size=3, dilation=8,
                               stride=1, padding=8)
        self.conv5 = nn.Conv2d(in_channels=in_channels, out_channels=8 * out_factor, kernel_size=3, dilation=16,
                               stride=1,
                               padding=16)

    def forward(self, x):
        bn = self.bn(x)
        conv1 = self.relu(self.conv1(bn))
        conv2 = self.relu(self.conv2(bn))
        conv3 = self.relu(self.conv3(bn))
        conv4 = self.relu(self.conv4(bn))
        conv5 = self.relu(self.conv5(bn))

        c5 = torch.cat([conv1, conv2, conv3, conv4, conv5], 1)

        return c5


class Unet(nn.Module):
    def downsample(self, in_channels, out_channels, kernel_size=3):
        block = nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )
        return block

    def conv_block(self, in_channels, mid_channel, out_channels, kernel_size=3, padding=1):
        block = nn.Sequential(
            nn.Conv2d(kernel_size=3, in_channels=in_channels, out_channels=mid_channel, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channel),
            nn.Conv2d(kernel_size=3, in_channels=mid_channel, out_channels=mid_channel, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channel),
            nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
        return block

    def __init__(self, in_channels, out_channels):
        super(Unet, self).__init__()
        self.downsample_1 = self.downsample(in_channels=in_channels, out_channels=64)

        self.downInception_1 = InceptionBlock(64, 1)  # output = 40
        self.maxpool_1 = nn.MaxPool2d(2, stride=2, return_indices=True)

        self.downInception_2 = InceptionBlock(40, 2)  # output = 80
        self.maxpool_2 = nn.MaxPool2d(2, stride=2, return_indices=True)

        self.downInception_3 = InceptionBlock(80, 3)  # output = 120
        self.maxpool_3 = nn.MaxPool2d(2, stride=2, return_indices=True)

        self.middleInception = InceptionBlock(120, 3)  # output = 120

        self.upsample_3 = nn.MaxUnpool2d(2, stride=2)
        self.upConvBlock_3 = self.conv_block(240, 120, 80)  # input = 120 + 120

        self.upsample_2 = nn.MaxUnpool2d(2, stride=2)
        self.upConvBlock_2 = self.conv_block(160, 80, 40)  # input = 80+80

        self.upsample_1 = nn.MaxUnpool2d(2, stride=2)
        self.upConvBlock_1 = self.conv_block(80, 40, 40)  # input = 40+40

        self.out = self.conv_block(40, 32, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        pre_dense = self.downsample_1(x)
        down_1 = self.downInception_1(pre_dense)
        pool_1, ind_1 = self.maxpool_1(down_1)

        down_2 = self.downInception_2(pool_1)
        pool_2, ind_2 = self.maxpool_2(down_2)

        down_3 = self.downInception_3(pool_2)
        pool_3, ind_3 = self.maxpool_3(down_3)

        middle = self.middleInception(pool_3)

        up_3 = self.upsample_3(middle, ind_3)
        up_3 = torch.cat([up_3, down_3], 1)
        up_3 = self.upConvBlock_3(up_3)

        up_2 = self.upsample_2(up_3, ind_2)
        up_2 = torch.cat([up_2, down_2], 1)
        up_2 = self.upConvBlock_2(up_2)

        up_1 = self.upsample_1(up_2, ind_1)
        up_1 = torch.cat([up_1, down_1], 1)
        up_1 = self.upConvBlock_1(up_1)

        output = self.out(up_1)

        return output
