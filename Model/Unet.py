import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils import autocrop

# 由于没有全连接层，所以不会对图片的尺寸有任何要求
# idea：放大，缩小图片是否会影响复原效果

class double_conv2d_bn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,stride=1,padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size,stride,padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size,stride,padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out

# 转置卷积一次


class deconv2d_bn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2,stride=2):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.upconv(x)))
        out = F.relu((self.upconv(x)))
        return out


class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        # 所有的卷积层
        self.layer1_conv = double_conv2d_bn(3, 64)
        self.layer2_conv = double_conv2d_bn(64, 128)
        self.layer3_conv = double_conv2d_bn(128, 256)
        self.layer4_conv = double_conv2d_bn(256, 512)
        self.layer5_conv = double_conv2d_bn(512, 1024)
        self.layer6_conv = double_conv2d_bn(1024, 512)
        self.layer7_conv = double_conv2d_bn(512, 256)
        self.layer8_conv = double_conv2d_bn(256, 128)
        self.layer9_conv = double_conv2d_bn(128, 64)
        self.layer10_conv = nn.Conv2d(64, 3, 1)
        # 所有的转置卷积层
        self.deconv1 = deconv2d_bn(1024, 512)
        self.deconv2 = deconv2d_bn(512, 256)
        self.deconv3 = deconv2d_bn(256, 128)
        self.deconv4 = deconv2d_bn(128, 64)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.layer1_conv(x)
        pool1 = F.max_pool2d(conv1, 2)

        conv2 = self.layer2_conv(pool1)
        pool2 = F.max_pool2d(conv2, 2)

        conv3 = self.layer3_conv(pool2)
        pool3 = F.max_pool2d(conv3, 2)

        conv4 = self.layer4_conv(pool3)
        pool4 = F.max_pool2d(conv4, 2)

        conv5 = self.layer5_conv(pool4)

        convt1 = self.deconv1(conv5)
        cropped_conv4, dec_layer = autocrop(conv4, convt1)
        concat1 = torch.cat([cropped_conv4, convt1], dim=1)
        conv6 = self.layer6_conv(concat1)

        convt2 = self.deconv2(conv6)
        cropped_conv3, dec_layer = autocrop(conv3, convt2)
        concat2 = torch.cat([cropped_conv3, convt2], dim=1)
        conv7 = self.layer7_conv(concat2)

        convt3 = self.deconv3(conv7)
        cropped_conv2, dec_layer = autocrop(conv2, convt3)
        concat3 = torch.cat([cropped_conv2, convt3], dim=1)
        conv8 = self.layer8_conv(concat3)

        convt4 = self.deconv4(conv8)
        cropped_conv1, dec_layer = autocrop(conv1, convt4)
        concat4 = torch.cat([cropped_conv1, convt4], dim=1)
        conv9 = self.layer9_conv(concat4)

        outp = self.layer10_conv(conv9)
        outp = self.sigmoid(outp)

        return outp


if __name__ == '__main__':
    import torch
    from torchvision import transforms
    from test import operate_pic

    test_image_path = r"C:\Users\guany\PycharmProjects\UDC\UDC_train\Train\Toled\LQ\1.png"
    label_image_path = r"C:\Users\guany\PycharmProjects\UDC\UDC_train\Train\Toled\HQ\1.png"

    model = Unet()
    input_tensor = operate_pic(test_image_path)

    outp_tensor = model(input_tensor)
    print(outp_tensor)
    print(outp_tensor.size())
