# 用于测算模型的PSNR等指标
import math

import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from Utils import Option
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
def operate_pic(test_image_path, crop_size=512):
    """
    返回可以输入model的裁剪后的图片tensor（4 dimensions）
    :param test_image_path: 图片路径
    :param crop_size: 裁剪尺寸
    :return: 可以输入model的裁剪后的图片tensor
    """
    # PIL读取图片
    img = Image.open(test_image_path).convert("RGB")
    # 转tensor，裁剪
    tran = transforms.Compose([
        transforms.ToTensor(),# 完成了归一化
        transforms.CenterCrop(crop_size)
    ])
    img_tensor = tran(img)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


if __name__ == '__main__':
    device = torch.device('cuda:0')
    test_image_path = r"C:\Users\guany\PycharmProjects\UDC\UDC_train\Train\Toled\LQ\1.png"
    label_image_path = r"C:\Users\guany\PycharmProjects\UDC\UDC_train\Train\Toled\HQ\1.png"
    # model_save_path = Option.DE_UNet_model_save_path
    model_save_path=Option.UNet_model_save_path
    unloader = transforms.ToPILImage()  # reconvert into PIL image

    model = torch.load(model_save_path)

    input_tensor = operate_pic(test_image_path)
    input_tensor = input_tensor.to(device)
    label_tensor = operate_pic(label_image_path)

    outp_tensor = model(input_tensor)
    criterion = torch.nn.L1Loss().cuda()

    outp_tensor = outp_tensor.squeeze(0)
    outp_mat = unloader(outp_tensor.cpu().detach())
    print(type(outp_mat))
    outp_mat = normalization(outp_mat)
    input_tensor = input_tensor.squeeze(0)
    intp_mat = unloader(input_tensor.cpu().detach())
    intp_mat = normalization(intp_mat)
    label_tensor = label_tensor.squeeze(0)
    label_mat = unloader(label_tensor.detach())
    label_mat = normalization(label_mat)

    # print(loss)
    print(type(outp_mat))
    print("预测图像与GT的PSNR对比 : ",psnr(outp_mat,label_mat,data_range=1))
    print("输入图像与GT的PSNR对比 : ",psnr(intp_mat, label_mat,data_range=1))
    print("预测图像与GT的SSIM对比 : ", ssim(outp_mat, label_mat, data_range=1,multichannel=True))
    print("输入图像与GT的SSIM对比 : ", ssim(intp_mat, label_mat,data_range=1,multichannel=True))
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(intp_mat)
    plt.title("Input")
    plt.subplot(1, 3, 2)
    plt.imshow(outp_mat)
    plt.title("Predict")
    plt.subplot(1, 3, 3)
    plt.imshow(label_mat)
    plt.title("GT")
    plt.show()

    # 总结
