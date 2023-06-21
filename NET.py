import torch.nn as nn
import torch
import torch.nn.functional as F

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        self.down1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=3,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=3,out_channels=3,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=3,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=3,out_channels=3,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.up1=nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.up2=nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.up3=nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.fc1=nn.Linear(3*28*28*2,3*28*28)
        self.fc2 = nn.Linear(3 * 56 * 56 * 2, 3 * 56 * 56)


    def forward(self,x):
        x=self.down1(x)
        d1=x

        x=self.down2(x)
        d2=x

        x=self.down3(x)
        d3=x

        x=torch.cat([x,d3],dim=0)
        x=x.view(-1)

        x=self.fc1(x)
        x=x.view(3,28,28)
        x=x.unsqueeze(0)
        x=self.up1(x)

        # print(x.shape, d3.shape)
        x=torch.cat([x,d2],dim=0)
        x=x.view(-1)
        x = self.fc2(x)
        x = x.view(3, 56, 56)
        x=x.unsqueeze(0)
        x=self.up2(x)

        x=self.up3(x)
        x=x.squeeze()
        # print(x.shape)
        return x


# testnet=MyNet()
#
# x=torch.ones((3,224,224))
# testnet(x)



