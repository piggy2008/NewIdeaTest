import torch
import torch.nn as nn
import torch.nn.functional as F
from underwater_model.op import *

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

class resblock_choice(nn.Module):
    def __init__(self, n_channels=128):
        super(resblock_choice, self).__init__()
        self.resblock_skipconnect = Identity(False)
        # self.resblock_sepconv = SepConv(n_channels, n_channels, 3, 1, 1, affine=True, upsample=False)
        # self.resblock_sepconvdouble = SepConvDouble(n_channels, n_channels, 3, 1, 1, affine=True, upsample=False)
        # self.resblock_dilconv = DilConv(n_channels, n_channels, 3, 1, 2, 2, affine=True, upsample=False)
        self.resblock_sge = sge_layer(16)
        self.resblock_dilconvdouble = DilConvDouble(n_channels, n_channels, 3, 1, 2, 2, affine=True, upsample=False)
        self.resblock_dil4Conv = DilConv(n_channels, n_channels, 3, 1, 4, 4, affine=True, upsample=False)
        self.resblock_conv = Conv(n_channels, n_channels, 3, 1, 1, affine=True, upsample=False)
        self.resblock_convdouble = ConvDouble(n_channels, n_channels, 3, 1, 1, affine=True, upsample=False)
        self.resblock_se = SELayer(n_channels)
        self.resblock_sa = sa_layer(n_channels, 16)


        # self.conv3 = nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, select):
        if select == 0:
            out = self.resblock_skipconnect(x)
        # elif select == 1:
        #     out = self.resblock_sepconv(x)
        # elif select == 2:
        #     out = self.resblock_sepconvdouble(x)
        elif select == 1:
            out = self.sge_layer(x)
        elif select == 2:
            out = self.resblock_dilconvdouble(x)
        elif select == 3:
            out = self.resblock_dil4Conv(x)
        elif select == 4:
            out = self.resblock_conv(x)
        elif select == 5:
            out = self.resblock_convdouble(x)
        elif select == 6:
            out = self.resblock_se(x)
        elif select == 7:
            out = self.resblock_sa(x)
        # out = self.conv3(out)
        return out + x


class Base(nn.Module):
    def __init__(self, channels):
        super(Base, self).__init__()
        self.conv1_hsv = nn.Sequential(nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=True))
        self.conv1_lab = nn.Sequential(nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=True))
        self.conv1_rgb = nn.Sequential(nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=True))

        self.conv2_hsv = nn.Sequential(nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=True))
        self.conv2_lab = nn.Sequential(nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=True))
        self.conv2_rgb = nn.Sequential(nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=True))

        self.conv3_hsv = nn.Sequential(nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=True))
        self.conv3_lab = nn.Sequential(nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=True))
        self.conv3_rgb = nn.Sequential(nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=True))

        self.block1 = resblock_choice(n_channels=channels[0])
        self.block2 = resblock_choice(n_channels=channels[0])
        self.block3 = resblock_choice(n_channels=channels[1])
        self.block4 = resblock_choice(n_channels=channels[1])
        self.block5 = resblock_choice(n_channels=channels[2])
        self.block6 = resblock_choice(n_channels=channels[2])

        # self.block1_rgb = resblock(n_channels=128)
        # self.block2_rgb = resblock(n_channels=128)
        # self.block3_rgb = resblock(n_channels=256)
        # self.block4_rgb = resblock(n_channels=256)
        # self.block5_rgb = resblock_choice(n_channels=512)
        # self.block6_rgb = resblock_choice(n_channels=512)

        # self.block1_lab = resblock(n_channels=128)
        # self.block2_lab = resblock(n_channels=128)
        # self.block3_lab = resblock(n_channels=256)
        # self.block4_lab = resblock(n_channels=256)
        # self.block5_lab = resblock_choice(n_channels=512)
        # self.block6_lab = resblock_choice(n_channels=512)

    def forward(self, rgb, hsv, lab, select):
        x_rgb = self.conv1_rgb(rgb)
        x_hsv = self.conv1_hsv(hsv)
        x_lab = self.conv1_lab(lab)
        # 128 * 128
        x_rgb = self.block1(x_rgb, select[0])
        x_rgb = self.block2(x_rgb, select[1])
        x_rgb2 = F.max_pool2d(x_rgb, kernel_size=3, stride=2, padding=1)
        x_rgb2 = self.conv2_rgb(x_rgb2)
        x_rgb2 = self.block3(x_rgb2, select[2])
        x_rgb2 = self.block4(x_rgb2, select[3])
        x_rgb3 = F.max_pool2d(x_rgb2, kernel_size=3, stride=2, padding=1)
        x_rgb3 = self.conv3_rgb(x_rgb3)
        x_rgb3 = self.block5(x_rgb3, select[4])
        x_rgb3 = self.block6(x_rgb3, select[5])
        # x_rgb3 = F.max_pool2d(x_rgb3, kernel_size=3, stride=2, padding=1)

        # 64 * 64
        x_hsv = self.block1(x_hsv, select[6])
        x_hsv = self.block2(x_hsv, select[7])
        x_hsv2 = F.max_pool2d(x_hsv, kernel_size=3, stride=2, padding=1)
        x_hsv2 = self.conv2_hsv(x_hsv2)
        x_hsv2 = self.block3(x_hsv2, select[8])
        x_hsv2 = self.block4(x_hsv2, select[9])
        x_hsv3 = F.max_pool2d(x_hsv2, kernel_size=3, stride=2, padding=1)
        x_hsv3 = self.conv3_hsv(x_hsv3)
        x_hsv3 = self.block5(x_hsv3, select[10])
        x_hsv3 = self.block6(x_hsv3, select[11])
        # x_hsv3 = F.max_pool2d(x_hsv3, kernel_size=3, stride=2, padding=1)

        # 64 * 64
        x_lab = self.block1(x_lab, select[12])
        x_lab = self.block2(x_lab, select[13])
        x_lab2 = F.max_pool2d(x_lab, kernel_size=3, stride=2, padding=1)
        x_lab2 = self.conv2_lab(x_lab2)
        x_lab2 = self.block3(x_lab2, select[14])
        x_lab2 = self.block4(x_lab2, select[15])
        x_lab3 = F.max_pool2d(x_lab2, kernel_size=3, stride=2, padding=1)
        x_lab3 = self.conv3_lab(x_lab3)
        x_lab3 = self.block5(x_lab3, select[16])
        x_lab3 = self.block6(x_lab3, select[17])
        return x_rgb, x_hsv, x_lab, x_rgb2, x_hsv2, x_lab2, \
               x_rgb3, x_hsv3, x_lab3