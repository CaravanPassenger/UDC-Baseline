from scipy.io import loadmat
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from Utils import Option
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# 读取mat文件，并从中取出图片ndarray
Toled_test_image_display = loadmat(
    r"C:\Users\guany\PycharmProjects\UDC\UDC_val_test\toled\toled_test_display.mat")
Toled_test_image_gt = loadmat(
    r"C:\Users\guany\PycharmProjects\UDC\UDC_val_test\toled\toled_test_gt.mat")
Toled_test_image_display = Toled_test_image_display['test_display']
Toled_test_image_gt = Toled_test_image_gt['test_gt']

# 对mat取出的ndarray进行预处理，可直接输入模型
def operate_pic(input, crop_size=64):
    """
    输入ndarray的nhwc图片集，返回经过{通道转换，归一化，中心裁剪}的tensor输入
    :param input:
    :param crop_size:
    :return:
    """
    img_tensor = torch.from_numpy(np.transpose(input,(0,3,1,2)))
    img_tensor = img_tensor.div(255)
    tran = transforms.Compose([
        transforms.CenterCrop(crop_size)
    ])
    img_tensor = tran(img_tensor)
    return img_tensor


def SSIM(intp_mat,gt_mat):
    sum_ssim=0
    for i in range(len(intp_mat)):
        input=np.transpose(intp_mat[i],(1,2,0))
        gt=np.transpose(gt_mat[i], (1,2,0))
        # plt.figure()
        # plt.imshow(input)
        # plt.show()
        sum_ssim+=ssim(input,gt, data_range=1, multichannel=True)
    return sum_ssim/len(intp_mat)


if __name__=="__main__":
    device = torch.device('cuda:0')
    DE_model = torch.load(Option.DE_UNet_model_save_path)
    UN_model = torch.load(Option.UNet_model_save_path)

    input_tensor = operate_pic(Toled_test_image_display)
    input_tensor = input_tensor.to(device)
    label_tensor = operate_pic(Toled_test_image_gt)

    outp_tensor1 = UN_model(input_tensor)
    outp_tensor2 = DE_model(input_tensor)

    outp1_mat = outp_tensor1.cpu().detach().numpy()
    outp2_mat = outp_tensor2.cpu().detach().numpy()
    intp_mat = input_tensor.cpu().detach().numpy()
    label_mat = label_tensor.numpy()
    # print(outp1_mat.max(), outp1_mat.min())
    # print(outp2_mat.max(), outp2_mat.min())
    # print(intp_mat.max(), intp_mat.min())
    # print(label_mat.max(), label_mat.min())

    in_psnr = round(psnr(intp_mat, label_mat, data_range=1), 4)
    in_ssim = round(SSIM(intp_mat, label_mat), 4)
    out1_psnr = round(psnr(outp1_mat, label_mat, data_range=1), 4)
    out1_ssim = round(SSIM(outp1_mat, label_mat), 4)
    out2_psnr = round(psnr(outp2_mat, label_mat, data_range=1), 4)
    out2_ssim = round(SSIM(outp2_mat, label_mat), 4)

    print("---Input---")
    print("PSNR: ",in_psnr)
    print("SSIM: ",in_ssim)
    print("---UNet---")
    print("UNet PSNR: ",out1_psnr)
    print("UNet SSIM: ",out1_ssim)
    print("---DE_UNet---")
    print("DE_UNet PSNR: ",out2_psnr)
    print("DE_UNet SSIM: ",out2_ssim)

# plt.figure()
# plt.subplot(1, 4, 1)
# plt.imshow(intp_mat)
# in_psnr = round(psnr(intp_mat, label_mat, data_range=1), 4)
# in_ssim = round(ssim(intp_mat, label_mat, data_range=1,multichannel=True), 4)
# plt.title("Input\nPSNR: " + str(in_psnr) + "\nSSIM: " + str(in_ssim))
# plt.subplot(1, 4, 2)
# plt.imshow(outp1_mat)
# out_psnr = round(psnr(outp1_mat, label_mat, data_range=1), 4)
# out_ssim = round(ssim(outp1_mat, label_mat, data_range=1,multichannel=True), 4)
# plt.title("UNet Predict\nPSNR: " + str(out_psnr) + "\nSSIM: " + str(out_ssim))
# plt.subplot(1, 4, 3)
# plt.imshow(outp2_mat)
# out_psnr = round(psnr(outp2_mat, label_mat, data_range=1), 4)
# out_ssim = round(ssim(outp2_mat, label_mat, data_range=1,multichannel=True), 4)
# plt.title("DE_UNet Predict\nPSNR: " + str(out_psnr) + "\nSSIM: " + str(out_ssim))
# plt.subplot(1, 4, 4)
# plt.imshow(label_mat)
# plt.title("GT")
# plt.show()
