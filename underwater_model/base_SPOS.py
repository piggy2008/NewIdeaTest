import torch
import torch.nn as nn
import torch.nn.functional as F
from underwater_model.op import *
from underwater_model.vit import ViT
from underwater_model.choice_block import block_choice

# top choice ] [1, 5, 5, 6, 5, 3, 7, 5, 4, 5, 2, 1, 6, 3, 7, 3]
class Base(nn.Module):
    def __init__(self, dim):
        super(Base, self).__init__()
        # self.conv1_hsv = nn.Sequential(nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1),
        #                                nn.ReLU(inplace=True))
        self.conv1_lab = nn.Sequential(nn.Conv2d(3, dim, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv1_rgb = nn.Sequential(nn.Conv2d(3, dim, kernel_size=3, stride=1, padding=1, bias=False))
        # self.conv2_hsv = nn.Sequential(nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1),
        #                                nn.ReLU(inplace=True))
        self.conv2_lab = nn.Sequential(nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.PixelUnshuffle(2))
        self.conv2_rgb = nn.Sequential(nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1),
                                       nn.PixelUnshuffle(2))

        # self.conv3_hsv = nn.Sequential(nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1),
        #                                nn.ReLU(inplace=True))
        self.conv3_lab = nn.Sequential(nn.Conv2d(int(dim*2**1), int(dim*2**1) // 2, kernel_size=3, stride=1, padding=1),
                                       nn.PixelUnshuffle(2))
        self.conv3_rgb = nn.Sequential(nn.Conv2d(int(dim*2**1), int(dim*2**1) // 2, kernel_size=3, stride=1, padding=1),
                                       nn.PixelUnshuffle(2))

        self.conv4_lab = nn.Sequential(
            nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 2) // 2, kernel_size=3, stride=1, padding=1),
            nn.PixelUnshuffle(2))
        self.conv4_rgb = nn.Sequential(
            nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 2) // 2, kernel_size=3, stride=1, padding=1),
            nn.PixelUnshuffle(2))



        self.block1 = block_choice(n_channels=dim)
        self.block2 = block_choice(n_channels=dim*2**1)
        self.block3 = block_choice(n_channels=dim*2**2)
        self.block4 = block_choice(n_channels=dim*2**3)

        # self.block1 = resblock_choice(n_channels=channels[0])
        # self.block2 = resblock_choice(n_channels=channels[0])
        # self.block3 = resblock_choice(n_channels=channels[1])
        # self.block4 = resblock_choice(n_channels=channels[1])
        # self.block5 = resblock_choice(n_channels=channels[2])
        # self.block6 = resblock_choice(n_channels=channels[2])

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

    def forward(self, rgb, lab, select):
        x_rgb = self.conv1_rgb(rgb)
        x_rgb = self.block1(x_rgb, select[0])

        x_rgb2 = self.conv2_rgb(x_rgb)
        x_rgb2 = self.block2(x_rgb2, select[1])

        x_rgb3 = self.conv3_rgb(x_rgb2)
        x_rgb3 = self.block3(x_rgb3, select[2])

        x_rgb4 = self.conv4_rgb(x_rgb3)
        x_rgb4 = self.block4(x_rgb4, select[3])

        # x_rgb3 = F.max_pool2d(x_rgb3, kernel_size=3, stride=2, padding=1)

        x_lab = self.conv1_lab(lab)
        x_lab = self.block1(x_lab, select[4])

        x_lab2 = self.conv2_lab(x_lab)
        x_lab2 = self.block2(x_lab2, select[5])

        x_lab3 = self.conv3_lab(x_lab2)
        x_lab3 = self.block3(x_lab3, select[6])

        x_lab4 = self.conv4_lab(x_lab3)
        x_lab4 = self.block4(x_lab4, select[7])

        # return x_rgb, x_lab, x_rgb2, x_lab2, \
        #        x_rgb3, x_lab3
        return x_rgb, x_rgb2, x_rgb3, x_rgb4, x_lab, x_lab2, x_lab3, x_lab4