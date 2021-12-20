import torch
import torch.nn as nn
import torch.nn.functional as F

class resblock(nn.Module):
    def __init__(self, n_channels=128):
        super(resblock, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        self.conv3 = nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return out + x


class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()
        self.conv1_hsv = nn.Sequential(nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=True))
        self.conv1_lab = nn.Sequential(nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=True))
        self.conv1_rgb = nn.Sequential(nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=True))

        self.conv2_hsv = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=True))
        self.conv2_lab = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=True))
        self.conv2_rgb = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=True))

        self.conv3_hsv = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=True))
        self.conv3_lab = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=True))
        self.conv3_rgb = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=True))

        self.block1_hsv = resblock(n_channels=128)
        self.block2_hsv = resblock(n_channels=128)
        self.block3_hsv = resblock(n_channels=256)
        self.block4_hsv = resblock(n_channels=256)
        self.block5_hsv = resblock(n_channels=512)
        self.block6_hsv = resblock(n_channels=512)

        self.block1_rgb = resblock(n_channels=128)
        self.block2_rgb = resblock(n_channels=128)
        self.block3_rgb = resblock(n_channels=256)
        self.block4_rgb = resblock(n_channels=256)
        self.block5_rgb = resblock(n_channels=512)
        self.block6_rgb = resblock(n_channels=512)

        self.block1_lab = resblock(n_channels=128)
        self.block2_lab = resblock(n_channels=128)
        self.block3_lab = resblock(n_channels=256)
        self.block4_lab = resblock(n_channels=256)
        self.block5_lab = resblock(n_channels=512)
        self.block6_lab = resblock(n_channels=512)

    def forward(self, rgb, hsv, lab):
        x_rgb = self.conv1_rgb(rgb)
        x_hsv = self.conv1_hsv(hsv)
        x_lab = self.conv1_lab(lab)
        # 128 * 128
        x_rgb = self.block1_rgb(x_rgb)
        x_rgb = self.block2_rgb(x_rgb)
        x_rgb2 = F.max_pool2d(x_rgb, kernel_size=3, stride=2, padding=1)
        x_rgb2 = self.conv2_rgb(x_rgb2)
        x_rgb2 = self.block3_rgb(x_rgb2)
        x_rgb2 = self.block4_rgb(x_rgb2)
        x_rgb3 = F.max_pool2d(x_rgb2, kernel_size=3, stride=2, padding=1)
        x_rgb3 = self.conv3_rgb(x_rgb3)
        x_rgb3 = self.block5_rgb(x_rgb3)
        x_rgb3 = self.block6_rgb(x_rgb3)
        # x_rgb3 = F.max_pool2d(x_rgb3, kernel_size=3, stride=2, padding=1)

        # 64 * 64
        x_hsv = self.block1_hsv(x_hsv)
        x_hsv = self.block2_hsv(x_hsv)
        x_hsv2 = F.max_pool2d(x_hsv, kernel_size=3, stride=2, padding=1)
        x_hsv2 = self.conv2_hsv(x_hsv2)
        x_hsv2 = self.block3_hsv(x_hsv2)
        x_hsv2 = self.block4_hsv(x_hsv2)
        x_hsv3 = F.max_pool2d(x_hsv2, kernel_size=3, stride=2, padding=1)
        x_hsv3 = self.conv3_hsv(x_hsv3)
        x_hsv3 = self.block5_hsv(x_hsv3)
        x_hsv3 = self.block6_hsv(x_hsv3)
        # x_hsv3 = F.max_pool2d(x_hsv3, kernel_size=3, stride=2, padding=1)

        # 64 * 64
        x_lab = self.block1_lab(x_lab)
        x_lab = self.block2_lab(x_lab)
        x_lab2 = F.max_pool2d(x_lab, kernel_size=3, stride=2, padding=1)
        x_lab2 = self.conv2_lab(x_lab2)
        x_lab2 = self.block3_lab(x_lab2)
        x_lab2 = self.block4_lab(x_lab2)
        x_lab3 = F.max_pool2d(x_lab2, kernel_size=3, stride=2, padding=1)
        x_lab3 = self.conv3_lab(x_lab3)
        x_lab3 = self.block5_lab(x_lab3)
        x_lab3 = self.block6_lab(x_lab3)
        return x_rgb, x_hsv, x_lab, x_rgb2, x_hsv2, x_lab2, \
               x_rgb3, x_hsv3, x_lab3