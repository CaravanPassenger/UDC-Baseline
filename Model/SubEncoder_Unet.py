import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils import autocrop

# 由于没有全连接层，所以不会对图片的尺寸有任何要求
# idea：放大，缩小图片是否会影响复原效果


class double_conv2d_bn(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size,
            stride,
            padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out

# 转置卷积一次


class deconv2d_bn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.upconv(x)))
        out = F.relu((self.upconv(x)))
        return out


class DE_Unet(nn.Module):
    def __init__(self):
        super().__init__()
        # 进来的卷积
        self.layer1_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        # subencoder1
        self.subencoder1 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            double_conv2d_bn(in_channels=32, out_channels=64),
            nn.MaxPool2d(2),
            double_conv2d_bn(in_channels=64, out_channels=128),
            nn.MaxPool2d(2),
            double_conv2d_bn(in_channels=128, out_channels=256),
            nn.MaxPool2d(2),
            double_conv2d_bn(in_channels=256, out_channels=512),
        )
        # subencoder2
        self.sub2_layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.sub2_layer2 = double_conv2d_bn(in_channels=32, out_channels=64)
        self.sub2_layer3 = double_conv2d_bn(in_channels=64, out_channels=128)
        self.sub2_layer4 = double_conv2d_bn(in_channels=128, out_channels=256)

        # decoder_conv
        self.decode_layer1_conv = double_conv2d_bn(
            in_channels=512, out_channels=256)
        self.decode_layer2_conv = double_conv2d_bn(
            in_channels=256, out_channels=128)
        self.decode_layer3_conv = double_conv2d_bn(
            in_channels=128, out_channels=64)
        self.decode_layer4_conv = double_conv2d_bn(
            in_channels=64, out_channels=32)
        # 所有的转置卷积层
        self.deconv1 = deconv2d_bn(512, 256)
        self.deconv2 = deconv2d_bn(256, 128)
        self.deconv3 = deconv2d_bn(128, 64)
        self.deconv4 = deconv2d_bn(64, 32)
        #
        self.final_operate = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1),
            # nn.BatchNorm2d(3),
            # nn.PixelShuffle(2)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1_conv(x)
        # sub1
        sub1_out = self.subencoder1(x)
        # sub2
        white4 = self.sub2_layer1(x)
        white3 = self.sub2_layer2(F.max_pool2d(white4, 2))
        white2 = self.sub2_layer3(F.max_pool2d(white3, 2))
        white1 = self.sub2_layer4(F.max_pool2d(white2, 2))
        # decoder
        blue1 = self.deconv1(sub1_out)
        cropped_white1, dec_layer = autocrop(white1, blue1)
        concat1 = torch.cat([cropped_white1, blue1], dim=1)
        blue2 = self.decode_layer1_conv(concat1)

        blue2 = self.deconv2(blue2)
        cropped_white2, dec_layer = autocrop(white2, blue2)
        concat2 = torch.cat([cropped_white2, blue2], dim=1)
        blue3 = self.decode_layer2_conv(concat2)

        blue3 = self.deconv3(blue3)
        cropped_white3, dec_layer = autocrop(white3, blue3)
        concat3 = torch.cat([cropped_white3, blue3], dim=1)
        blue4 = self.decode_layer3_conv(concat3)

        blue4 = self.deconv4(blue4)
        cropped_white4, dec_layer = autocrop(white4, blue4)
        concat4 = torch.cat([cropped_white4, blue4], dim=1)
        blue5 = self.decode_layer4_conv(concat4)

        outp = self.final_operate(blue5)
        outp = self.sigmoid(outp)
        return outp
