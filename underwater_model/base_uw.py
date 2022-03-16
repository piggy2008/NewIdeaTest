import torch
import torch.nn as nn
import torch.nn.functional as F
from underwater_model.op import *
from underwater_model.restormer import TransformerBlock
from underwater_model.trans_block_dual import TransformerBlock_dual
from underwater_model.trans_block_eca import TransformerBlock_eca
from underwater_model.trans_block_sa import TransformerBlock_sa
from underwater_model.trans_block_sge import TransformerBlock_sge

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



        self.block1_rgb = DoubleAttentionLayer(dim, dim, dim)
        self.block2_rgb = nn.Sequential(
            *[TransformerBlock_dual(dim=int(dim*2**1), num_heads=2, ffn_expansion_factor=2.66,
                               bias=False, LayerNorm_type='WithBias') for i in range(1)])
        self.block3_rgb = nn.Sequential(
            *[TransformerBlock_dual(dim=int(dim*2**2), num_heads=2, ffn_expansion_factor=2.66,
                               bias=False, LayerNorm_type='WithBias') for i in range(1)])
        self.block4_rgb = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=2, ffn_expansion_factor=2.66,
                         bias=False, LayerNorm_type='WithBias') for i in range(1)])

        self.block1_lab = SELayer(dim)
        self.block2_lab = nn.Sequential(
            *[TransformerBlock_sge(dim=int(dim * 2 ** 1), num_heads=2, ffn_expansion_factor=2.66,
                                    bias=False, LayerNorm_type='WithBias') for i in range(1)])
        self.block3_lab = nn.Sequential(
            *[TransformerBlock_sge(dim=int(dim * 2 ** 2), num_heads=2, ffn_expansion_factor=2.66,
                                    bias=False, LayerNorm_type='WithBias') for i in range(1)])
        self.block4_lab = DilConv(dim * 2 ** 3, dim * 2 ** 3, 3, 1, 4, 4, affine=True, upsample=False)
    def forward(self, rgb, lab):
        x_rgb = self.conv1_rgb(rgb)
        x_rgb = self.block1_rgb(x_rgb) + x_rgb
        # x_rgb = self.block1_2(x_rgb, select[1])

        x_rgb2 = self.conv2_rgb(x_rgb)
        x_rgb2 = self.block2_rgb(x_rgb2) + x_rgb2
        # x_rgb2 = self.block2_2(x_rgb2, select[3])

        x_rgb3 = self.conv3_rgb(x_rgb2)
        x_rgb3 = self.block3_rgb(x_rgb3) + x_rgb3
        # x_rgb3 = self.block3_2(x_rgb3, select[5])

        x_rgb4 = self.conv4_rgb(x_rgb3)
        x_rgb4 = self.block4_rgb(x_rgb4) + x_rgb4

        # x_rgb3 = F.max_pool2d(x_rgb3, kernel_size=3, stride=2, padding=1)

        x_lab = self.conv1_lab(lab)
        x_lab = self.block1_lab(x_lab) + x_lab
        # x_lab = self.block1_2(x_lab, select[8])

        x_lab2 = self.conv2_lab(x_lab)
        x_lab2 = self.block2_lab(x_lab2) + x_lab2
        # x_lab2 = self.block2_2(x_lab2, select[10])

        x_lab3 = self.conv3_lab(x_lab2)
        x_lab3 = self.block3_lab(x_lab3) + x_lab3
        # x_lab3 = self.block3_2(x_lab3, select[12])

        x_lab4 = self.conv4_lab(x_lab3)
        x_lab4 = self.block4_lab(x_lab4) + x_lab4

        # return x_rgb, x_lab, x_rgb2, x_lab2, \
        #        x_rgb3, x_lab3
        return x_rgb, x_rgb2, x_rgb3, x_rgb4, x_lab, x_lab2, x_lab3, x_lab4