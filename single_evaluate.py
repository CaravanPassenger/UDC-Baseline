from scipy.io import loadmat
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from Utils import Option
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

Toled_test_image_display = loadmat(
    r"C:\Users\guany\PycharmProjects\UDC\UDC_val_test\toled\toled_test_display.mat")
Toled_test_image_gt = loadmat(
    r"C:\Users\guany\PycharmProjects\UDC\UDC_val_test\toled\toled_test_gt.mat")
Toled_test_image_display = Toled_test_image_display['test_display']
Toled_test_image_gt = Toled_test_image_gt['test_gt']

def operate_pic(input_set,epoch,crop_size=256):
    """
    输入ndarray的nhwc图片集，返回经过{通道转换，归一化，中心裁剪}的tensor输入
    :param input:
    :param crop_size:
    :return:
    """
    input=input_set[epoch]
    img_tensor = torch.from_numpy(np.transpose(input,(2,0,1)))
    img_tensor = img_tensor.div(255)
    tran = transforms.Compose([
        transforms.CenterCrop(crop_size)
    ])
    img_tensor = tran(img_tensor)
    img_tensor=img_tensor.unsqueeze(0)
    return img_tensor

if __name__=="__main__":
    device = torch.device('cuda:0')
    DE_model = torch.load(Option.DE_UNet_model_save_path)
    UN_model = torch.load(Option.UNet_model_save_path)
    for i in range(25,30):
        input_tensor = operate_pic(Toled_test_image_display,epoch=i)
        input_tensor = input_tensor.to(device)
        label_tensor = operate_pic(Toled_test_image_gt,epoch=i)

        outp_tensor1 = UN_model(input_tensor)
        outp_tensor2 = DE_model(input_tensor)

        intp_mat = input_tensor.cpu().detach().squeeze(0).numpy()
        outp1_mat = outp_tensor1.cpu().detach().squeeze(0).numpy()
        outp2_mat = outp_tensor2.cpu().detach().squeeze(0).numpy()
        label_mat = label_tensor.numpy().squeeze(0)

        intp_mat=np.transpose(intp_mat,(1,2,0))
        outp1_mat=np.transpose(outp1_mat,(1,2,0))
        outp2_mat=np.transpose(outp2_mat,(1,2,0))
        label_mat=np.transpose(label_mat,(1,2,0))

        plt.figure()
        plt.subplot(1, 4, 1)
        plt.axis('off')
        plt.imshow(intp_mat)
        in_psnr = round(psnr(intp_mat, label_mat, data_range=1), 4)
        in_ssim = round(ssim(intp_mat, label_mat, data_range=1,multichannel=True), 4)
        plt.title("Input\nPSNR: " + str(in_psnr) + "\nSSIM: " + str(in_ssim))
        plt.subplot(1, 4, 2)
        plt.axis('off')
        plt.imshow(outp1_mat)
        out_psnr = round(psnr(outp1_mat, label_mat, data_range=1), 4)
        out_ssim = round(ssim(outp1_mat, label_mat, data_range=1,multichannel=True), 4)
        plt.title("UNet Predict\nPSNR: " + str(out_psnr) + "\nSSIM: " + str(out_ssim))
        plt.subplot(1, 4, 3)
        plt.axis('off')
        plt.imshow(outp2_mat)
        out_psnr = round(psnr(outp2_mat, label_mat, data_range=1), 4)
        out_ssim = round(ssim(outp2_mat, label_mat, data_range=1,multichannel=True), 4)
        plt.title("DE_UNet Predict\nPSNR: " + str(out_psnr) + "\nSSIM: " + str(out_ssim))
        plt.subplot(1, 4, 4)
        plt.axis('off')
        plt.imshow(label_mat)
        plt.title("GT")

        plt.savefig(r"C:\Users\guany\PycharmProjects\UDC\Toled_test_imgs\Patch256"+"\\"+str(i),dpi=300,bbox_inches = 'tight')
        # plt.show()