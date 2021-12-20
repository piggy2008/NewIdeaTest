import torch
import torch.nn as nn
import torch.nn.functional as F
from underwater_model.op import *

class Search(nn.Module):
    def __init__(self):
        super(Search, self).__init__()
        # round 1(1~3), level 1(1~3), op:1~12
        self.r1_l1_zero = Zero(1, True)
        self.r1_l1_skipconnect = Identity(True)
        self.r1_l1_sepconv = SepConv(128, 128, 3, 1, 1, affine=True, upsample=True)
        self.r1_l1_sepconvdouble = SepConvDouble(128, 128, 3, 1, 1, affine=True, upsample=True)
        self.r1_l1_dilconv = DilConv(128, 128, 3, 1, 2, 2, affine=True, upsample=True)
        self.r1_l1_dilconvdouble = DilConvDouble(128, 128, 3, 1, 2, 2, affine=True, upsample=True)
        self.r1_l1_dil4Conv = DilConv(128, 128, 3, 1, 4, 4, affine=True, upsample=True)
        self.r1_l1_conv = Conv(128, 128, 3, 1, 1, affine=True, upsample=True)
        self.r1_l1_convdouble = ConvDouble(128, 128, 3, 1, 1, affine=True, upsample=True)
        # self.r1_l1_sa = SpatialAttention(128, 7)
        # self.r1_l1_ca = ChannelAttention(128, 16)
        # self.r1_l1_se = SELayer(128)

        self.r1_l13_zero = Zero(1, True)
        self.r1_l13_skipconnect = Identity(True)
        self.r1_l13_sepconv = SepConv(128, 128, 3, 1, 1, affine=True, upsample=True)
        self.r1_l13_sepconvdouble = SepConvDouble(128, 128, 3, 1, 1, affine=True, upsample=True)
        self.r1_l13_dilconv = DilConv(128, 128, 3, 1, 2, 2, affine=True, upsample=True)
        self.r1_l13_dilconvdouble = DilConvDouble(128, 128, 3, 1, 2, 2, affine=True, upsample=True)
        self.r1_l13_dil4Conv = DilConv(128, 128, 3, 1, 4, 4, affine=True, upsample=True)
        self.r1_l13_conv = Conv(128, 128, 3, 1, 1, affine=True, upsample=True)
        self.r1_l13_convdouble = ConvDouble(128, 128, 3, 1, 1, affine=True, upsample=True)
        # self.r1_l13_se = SELayer(128)

        self.r1_l2_zero = Zero(1, True)
        self.r1_l2_skipconnect = Identity(True)
        self.r1_l2_sepconv = SepConv(128, 128, 3, 1, 1, affine=True, upsample=True)
        self.r1_l2_sepconvdouble = SepConvDouble(128, 128, 3, 1, 1, affine=True, upsample=True)
        self.r1_l2_dilconv = DilConv(128, 128, 3, 1, 2, 2, affine=True, upsample=True)
        self.r1_l2_dilconvdouble = DilConvDouble(128, 128, 3, 1, 2, 2, affine=True, upsample=True)
        self.r1_l2_dil4Conv = DilConv(128, 128, 3, 1, 4, 4, affine=True, upsample=True)
        self.r1_l2_conv = Conv(128, 128, 3, 1, 1, affine=True, upsample=True)
        self.r1_l2_convdouble = ConvDouble(128, 128, 3, 1, 1, affine=True, upsample=True)
        # self.r1_l2_sa = SpatialAttention(128, 7)
        # self.r1_l2_ca = ChannelAttention(128, 16)
        # self.r1_l2_se = SELayer(128)

        self.r1_l3_zero = Zero(1, False)
        self.r1_l3_skipconnect = Identity(False)
        self.r1_l3_sepconv = SepConv(128, 128, 3, 1, 1, affine=True, upsample=False)
        self.r1_l3_sepconvdouble = SepConvDouble(128, 128, 3, 1, 1, affine=True, upsample=False)
        self.r1_l3_dilconv = DilConv(128, 128, 3, 1, 2, 2, affine=True, upsample=False)
        self.r1_l3_dilconvdouble = DilConvDouble(128, 128, 3, 1, 2, 2, affine=True, upsample=False)
        self.r1_l3_dil4Conv = DilConv(128, 128, 3, 1, 4, 4, affine=True, upsample=False)
        self.r1_l3_conv = Conv(128, 128, 3, 1, 1, affine=True, upsample=False)
        self.r1_l3_convdouble = ConvDouble(128, 128, 3, 1, 1, affine=True, upsample=False)
        # self.r1_l3_se = SELayer(128)
        # self.r1_l3_sa = SpatialAttention(128, 7)
        # self.r1_l3_ca = ChannelAttention(128, 16)

    def forward(self, x, y, z, select):
        if select[0] == 0:
            xy = self.r1_l1_zero(x)
        elif select[0] == 1:
            xy = self.r1_l1_skipconnect(x)
        elif select[0] == 2:
            xy = self.r1_l1_sepconv(x)
        elif select[0] == 3:
            xy = self.r1_l1_sepconvdouble(x)
        elif select[0] == 4:
            xy = self.r1_l1_dilconv(x)
        elif select[0] == 5:
            xy = self.r1_l1_dilconvdouble(x)
        elif select[0] == 6:
            xy = self.r1_l1_dil4Conv(x)
        elif select[0] == 7:
            xy = self.r1_l1_conv(x)
        elif select[0] == 8:
            xy = self.r1_l1_convdouble(x)
        # elif select[0] == 9:
        #     xy = self.r1_l1_se(x)
        # else:
        #     x = self.r1_l1_ca(x)
        if xy.size()[2:] != y.size()[2:]:
            xy = F.interpolate(xy, y.size()[2:], mode='bilinear')

        y = y + xy
        if select[1] == 0:
            yz = self.r1_l2_zero(y)
        elif select[1] == 1:
            yz = self.r1_l2_skipconnect(y)
        elif select[1] == 2:
            yz = self.r1_l2_sepconv(y)
        elif select[1] == 3:
            yz = self.r1_l2_sepconvdouble(y)
        elif select[1] == 4:
            yz = self.r1_l2_dilconv(y)
        elif select[1] == 5:
            yz = self.r1_l2_dilconvdouble(y)
        elif select[1] == 6:
            yz = self.r1_l2_dil4Conv(y)
        elif select[1] == 7:
            yz = self.r1_l2_conv(y)
        elif select[1] == 8:
            yz = self.r1_l2_convdouble(y)
        # elif select[1] == 9:
        #     yz = self.r1_l2_se(y)
        # else:
        #     y = self.r1_l2_ca(y)

        if yz.size()[2:] != z.size()[2:]:
            yz = F.interpolate(yz, z.size()[2:], mode='bilinear')

        if select[3] == 0:
            xz = self.r1_l13_zero(x)
        elif select[3] == 1:
            xz = self.r1_l13_skipconnect(x)
        elif select[3] == 2:
            xz = self.r1_l13_sepconv(x)
        elif select[3] == 3:
            xz = self.r1_l13_sepconvdouble(x)
        elif select[3] == 4:
            xz = self.r1_l13_dilconv(x)
        elif select[3] == 5:
            xz = self.r1_l13_dilconvdouble(x)
        elif select[3] == 6:
            xz = self.r1_l13_dil4Conv(x)
        elif select[3] == 7:
            xz = self.r1_l13_conv(x)
        elif select[3] == 8:
            xz = self.r1_l13_convdouble(x)
        # elif select[3] == 9:
        #     xz = self.r1_l1_se(x)
        # else:
        #     x = self.r1_l1_ca(x)
        if xz.size()[2:] != z.size()[2:]:
            xz = F.interpolate(xz, z.size()[2:], mode='bilinear')

        z = z + yz + xz

        if select[2] == 0:
            z = self.r1_l3_zero(z)
        elif select[2] == 1:
            z = self.r1_l3_skipconnect(z)
        elif select[2] == 2:
            z = self.r1_l3_sepconv(z)
        elif select[2] == 3:
            z = self.r1_l3_sepconvdouble(z)
        elif select[2] == 4:
            z = self.r1_l3_dilconv(z)
        elif select[2] == 5:
            z = self.r1_l3_dilconvdouble(z)
        elif select[2] == 6:
            z = self.r1_l3_dil4Conv(z)
        elif select[2] == 7:
            z = self.r1_l3_conv(z)
        elif select[2] == 8:
            z = self.r1_l3_convdouble(z)
        # elif select[2] == 9:
        #     z = self.r1_l3_se(z)
        # else:
        #     z = self.r1_l3_ca(z)
        fuse = z + xz + yz

        return fuse

if __name__ == '__main__':
    x = torch.zeros(2, 128, 32, 32)
    y = torch.zeros(2, 128, 64, 64)
    z = torch.zeros(2, 128, 128, 128)

    model = Search()
    choice = [9, 9, 9, 9]
    output = model(x, y, z, choice)
    print(output.size())
