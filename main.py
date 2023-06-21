import torch
from torch import nn
from torch import optim
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import NET
import tqdm
import sys
import unet_2

# 创建一个变换对象，用于将图像缩放为指定大小并居中裁剪
resize_crop = transforms.Compose([
    transforms.Resize(256),  # 缩放图像，使其短边不小于256像素
    transforms.CenterCrop(224)  # 居中裁剪图像，使其长和宽都等于224像素
])

# 创建一个变换对象，用于将图像转换为Tensor格式并进行归一化处理
to_tensor_normalize = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为Tensor格式
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化处理
])

to_tensor_mask = transforms.Compose([
    transforms.Resize(256),  # 缩放图像，使其短边不小于256像素
    transforms.CenterCrop(224),  # 居中裁剪图像，使其长和宽都等于224像素
    transforms.ToTensor()  # 将图像转换为Tensor格式
])

class MyDataset(Dataset):

    def __init__(self,root_dir):
        self.root_dir=root_dir
        self.img_root=root_dir + r"\last"
        self.mask_root = root_dir + r"\last_msk"
        self.img_names=os.listdir(self.img_root)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path=os.path.join(self.img_root,self.img_names[idx])
        mask_path = os.path.join(self.mask_root, self.img_names[idx])

        img=Image.open(img_path)
        img = resize_crop(img)
        img_tensor = to_tensor_normalize(img)

        mask=Image.open(mask_path)
        mask_tensor=to_tensor_mask(mask).squeeze()
        return img_tensor,mask_tensor

    def test_showdir(self):
        print(self.root_dir)

    def test_showitem(self,idx):
        img_path = os.path.join(self.img_root, self.img_names[idx])
        mat=cv2.imread(img_path,1)
        cv2.imshow("test_showitem",mat)
        # label = self.img_names[idx][:-4]
        # print("label:",label)
        print(self.__getitem__(idx)[0].shape)
        print(self.__getitem__(idx)[1].shape)
        cv2.waitKey(0)


def train(epochs):
    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('正在使用{}个线程加载数据集'.format(nw))
    train_dataset=MyDataset(r"D:\ML_DATA\handbag\train")
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"找到 {device_count} 一个CUDA 设备:", end='')
        for i in range(device_count):
            print(torch.cuda.get_device_name(i))
    else:
        print("没有找到CUDA设备")
    print("device:", device)

    MyNet = unet_2.Unet()
    MyNet.to(device)
    # 将数据移动至显卡（cuda）
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(MyNet.parameters(), lr=0.001)
    save_path = r'.\UNET.pth'
    print("权重文件保存路径：", os.path.join(os.getcwd()))

    # epochs = 100
    best_acc = 0.0
    train_steps = len(train_loader)
    MyNet.train()
    for epoch in range(epochs):
        print(epoch)
        running_loss = 0
        # train_bar = tqdm(train_loader, file=sys.stdout)
        # 此行代码用于将可迭代对象的迭代过程转化为进度条并输出到控制台
        for data in train_loader:
            images, mask = data
            # print(images.shape, mask.shape)
            optimizer.zero_grad()
            outputs = MyNet(images.to(device)).squeeze()
            # print(outputs.shape,mask.shape)
            # exit(2)
            # mask=mask.squeeze()
            loss = loss_function(outputs, mask.to(device))
            # 将数据移动至显卡（cuda）
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
            #                                                          epochs,
            #                                                          loss)
        torch.save(MyNet.state_dict(), "end.pth")
        print(running_loss)
    print('Finished Training')

if __name__ == '__main__':
    train(1000)

# testdataset=MyDataset(r"D:\ML_DATA\handbag\train")
# testdataset.test_showitem(1)

# img = Image.open(r"D:\ML_DATA\handbag\train\last\0.jpg")
# img = resize_crop(img)
# img = to_tensor_normalize(img)
#
# # 将Tensor图像转换为NumPy数组，并移动数据到CPU上
# img = img.cpu().numpy().transpose((1, 2, 0))
#
# # 显示处理后的图像
# plt.imshow(img)
# plt.show()