import torch
import unet_2
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def binary_arr(arr, threshold):
    """
    将输入的numpy数组按照指定的阈值threshold进行二值化
    :param arr: 输入的numpy数组
    :param threshold: 阈值
    :return: 二值化后的numpy数组
    """
    binary_arr = np.zeros_like(arr)  # 创建一个和arr大小相同的全零数组
    binary_arr[arr > threshold] = 255  # 将大于阈值的元素赋值为1
    return binary_arr

testnet=unet_2.Unet()

testnet.load_state_dict(torch.load('end.pth'))

resize_crop = transforms.Compose([
    transforms.Resize(256),  # 缩放图像，使其短边不小于256像素
    transforms.CenterCrop(224)  # 居中裁剪图像，使其长和宽都等于224像素
])

to_tensor_normalize = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为Tensor格式
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化处理
])

img_path=r"D:\pythonItem\语义分割\test_data\01.jpg"

img = Image.open(img_path)
img = resize_crop(img)
img_tensor = to_tensor_normalize(img).unsqueeze(0)

testnet.eval()
with torch.no_grad():

    out=testnet(img_tensor).squeeze()


    out=out*255
    img=out.numpy()

    img=binary_arr(img,125)
    plt.imshow(img, cmap='gray')  # 绘制图像，cmap表示使用灰度颜色映射
    plt.show()  # 显示图像
    # print(out.shape)
    # print(out)
    # img = transforms.ToPILImage(out)
    # # print(img)
    # plt.imshow(img)
    # plt.show()

