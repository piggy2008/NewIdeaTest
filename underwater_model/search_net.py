import torch
import torch.nn as nn
import torch.nn.functional as F
from underwater_model.op import *

class Round1(nn.Module):
    def __init__(self, channel=128):
        super(Round1, self).__init__()
        # self.r1_l1_zero = Zero(1, True)
        self.r1_l1_skipconnect = Identity(False)
        # self.r1_l1_sepconv = SepConv(channel, channel, 3, 1, 1, affine=True, upsample=True)
        # self.r1_l1_sepconvdouble = SepConvDouble(channel, channel, 3, 1, 1, affine=True, upsample=True)
        # self.r1_l1_dilconv = DilConv(channel, channel, 3, 1, 2, 2, affine=True, upsample=True)
        self.r1_l1_sge = sge_layer(16)
        # self.r1_l1_dilconvdouble = DilConvDouble(channel, channel, 3, 1, 2, 2, affine=True, upsample=True)
        self.r1_l1_eca = eca_layer(3)
        self.r1_l1_dil4Conv = DilConv(channel, channel, 3, 1, 4, 4, affine=True, upsample=False)
        self.r1_l1_conv = Conv(channel, channel, 3, 1, 1, affine=True, upsample=False)
        self.r1_l1_da = DoubleAttentionLayer(channel, channel, channel)
        self.r1_l1_se = SELayer(channel)
        self.r1_l1_sa = sa_layer(channel, groups=16)

        # self.r1_l13_zero = Zero(1, True)
        self.r1_l13_skipconnect = Identity(False)
        # self.r1_l13_sepconv = SepConv(channel, channel, 3, 1, 1, affine=True, upsample=True)
        # self.r1_l13_sepconvdouble = SepConvDouble(channel, channel, 3, 1, 1, affine=True, upsample=True)
        # self.r1_l13_dilconv = DilConv(channel, channel, 3, 1, 2, 2, affine=True, upsample=True)
        self.r1_l13_sge = sge_layer(16)
        # self.r1_l13_dilconvdouble = DilConvDouble(channel, channel, 3, 1, 2, 2, affine=True, upsample=True)
        self.r1_l13_eca = eca_layer(3)
        self.r1_l13_dil4Conv = DilConv(channel, channel, 3, 1, 4, 4, affine=True, upsample=False)
        self.r1_l13_conv = Conv(channel, channel, 3, 1, 1, affine=True, upsample=False)
        self.r1_l13_da = DoubleAttentionLayer(channel, channel, channel)
        self.r1_l13_se = SELayer(channel)
        self.r1_l13_sa = sa_layer(channel, groups=16)

        # self.r1_l2_zero = Zero(1, True)
        self.r1_l2_skipconnect = Identity(False)
        # self.r1_l2_sepconv = SepConv(channel, channel, 3, 1, 1, affine=True, upsample=True)
        # self.r1_l2_sepconvdouble = SepConvDouble(channel, channel, 3, 1, 1, affine=True, upsample=True)
        # self.r1_l2_dilconv = DilConv(channel, channel, 3, 1, 2, 2, affine=True, upsample=True)
        self.r1_l2_sge = sge_layer(16)
        # self.r1_l2_dilconvdouble = DilConvDouble(channel, channel, 3, 1, 2, 2, affine=True, upsample=True)
        self.r1_l2_eca = eca_layer(3)
        self.r1_l2_dil4Conv = DilConv(channel, channel, 3, 1, 4, 4, affine=True, upsample=False)
        self.r1_l2_conv = Conv(channel, channel, 3, 1, 1, affine=True, upsample=False)
        self.r1_l2_da = DoubleAttentionLayer(channel, channel, channel)
        self.r1_l2_se = SELayer(channel)
        self.r1_l2_sa = sa_layer(channel, groups=16)

        # self.r1_l3_zero = Zero(1, False)
        self.r1_l3_skipconnect = Identity(False)
        # self.r1_l3_sepconv = SepConv(channel, channel, 3, 1, 1, affine=True, upsample=False)
        # self.r1_l3_sepconvdouble = SepConvDouble(channel, channel, 3, 1, 1, affine=True, upsample=False)
        # self.r1_l3_dilconv = DilConv(channel, channel, 3, 1, 2, 2, affine=True, upsample=False)
        self.r1_l3_sge = sge_layer(16)
        # self.r1_l3_dilconvdouble = DilConvDouble(channel, channel, 3, 1, 2, 2, affine=True, upsample=False)
        self.r1_l3_eca = eca_layer(3)
        self.r1_l3_dil4Conv = DilConv(channel, channel, 3, 1, 4, 4, affine=True, upsample=False)
        self.r1_l3_conv = Conv(channel, channel, 3, 1, 1, affine=True, upsample=False)
        self.r1_l3_da = DoubleAttentionLayer(channel, channel, channel)
        self.r1_l3_se = SELayer(channel)
        self.r1_l3_sa = sa_layer(channel, groups=16)

    def forward(self, x, y, z, select):
        # if select[0] == 0:
        #     xy = self.r1_l1_zero(x)
        if select[0] == 0:
            xy = self.r1_l1_skipconnect(x)
        # elif select[0] == 1:
        #     xy = self.r1_l1_sepconv(x)
        # elif select[0] == 2:
        #     xy = self.r1_l1_sepconvdouble(x)
        elif select[0] == 1:
            xy = self.r1_l1_sge(x)
        elif select[0] == 2:
            xy = self.r1_l1_eca(x)
        elif select[0] == 3:
            xy = self.r1_l1_dil4Conv(x)
        elif select[0] == 4:
            xy = self.r1_l1_conv(x)
        elif select[0] == 5:
            xy = self.r1_l1_da(x)
        elif select[0] == 6:
            xy = self.r1_l1_se(x)
        elif select[0] == 7:
            xy = self.r1_l1_sa(x)
        # else:
        #     x = self.r1_l1_ca(x)
        xy_temp = xy
        if xy.size()[2:] != y.size()[2:]:
            xy = F.interpolate(xy, y.size()[2:], mode='bilinear')

        y = y + xy
        # if select[1] == 0:
        #     yz = self.r1_l2_zero(y)
        if select[1] == 0:
            yz = self.r1_l2_skipconnect(y)
        # elif select[1] == 1:
        #     yz = self.r1_l2_sepconv(y)
        # elif select[1] == 2:
        #     yz = self.r1_l2_sepconvdouble(y)
        elif select[1] == 1:
            yz = self.r1_l2_sge(y)
        elif select[1] == 2:
            yz = self.r1_l2_eca(y)
        elif select[1] == 3:
            yz = self.r1_l2_dil4Conv(y)
        elif select[1] == 4:
            yz = self.r1_l2_conv(y)
        elif select[1] == 5:
            yz = self.r1_l2_da(y)
        elif select[1] == 6:
            yz = self.r1_l2_se(y)
        elif select[1] == 7:
            yz = self.r1_l2_sa(y)
        # else:
        #     y = self.r1_l2_ca(y)
        yz_temp = yz
        if yz.size()[2:] != z.size()[2:]:
            yz = F.interpolate(yz, z.size()[2:], mode='bilinear')
        #
        # if select[3] == 0:
        #     xz = self.r1_l13_zero(x)
        if select[3] == 0:
            xz = self.r1_l13_skipconnect(x)
        # elif select[3] == 1:
        #     xz = self.r1_l13_sepconv(x)
        # elif select[3] == 2:
        #     xz = self.r1_l13_sepconvdouble(x)
        elif select[3] == 1:
            xz = self.r1_l13_sge(x)
        elif select[3] == 2:
            xz = self.r1_l13_eca(x)
        elif select[3] == 3:
            xz = self.r1_l13_dil4Conv(x)
        elif select[3] == 4:
            xz = self.r1_l13_conv(x)
        elif select[3] == 5:
            xz = self.r1_l13_da(x)
        elif select[3] == 6:
            xz = self.r1_l13_se(x)
        elif select[3] == 7:
            xz = self.r1_l13_sa(x)
        # else:
        #     x = self.r1_l1_ca(x)
        xz_temp = xz
        if xz.size()[2:] != z.size()[2:]:
            xz = F.interpolate(xz, z.size()[2:], mode='bilinear')

        z = z + yz + xz

        # if select[2] == 0:
        #     z = self.r1_l3_zero(z)
        if select[2] == 0:
            z = self.r1_l3_skipconnect(z)
        # elif select[2] == 1:
        #     z = self.r1_l3_sepconv(z)
        # elif select[2] == 2:
        #     z = self.r1_l3_sepconvdouble(z)
        elif select[2] == 1:
            z = self.r1_l3_sge(z)
        elif select[2] == 2:
            z = self.r1_l3_eca(z)
        elif select[2] == 3:
            z = self.r1_l3_dil4Conv(z)
        elif select[2] == 4:
            z = self.r1_l3_conv(z)
        elif select[2] == 5:
            z = self.r1_l3_da(z)
        elif select[2] == 6:
            z = self.r1_l3_se(z)
        elif select[2] == 7:
            z = self.r1_l3_sa(z)
        # else:
        #     z = self.r1_l3_ca(z)
        z_temp = z
        fuse = z + xz + yz

        return fuse, xy_temp, yz_temp, z_temp

class Round1_chosen(nn.Module):
    def __init__(self, channel=128):
        super(Round1_chosen, self).__init__()
        # self.r1_l1_zero = Zero(1, True)
        # self.r1_l1_skipconnect = Identity(False)
        # self.r1_l1_sepconv = SepConv(channel, channel, 3, 1, 1, affine=True, upsample=True)
        # self.r1_l1_sepconvdouble = SepConvDouble(channel, channel, 3, 1, 1, affine=True, upsample=True)
        # self.r1_l1_dilconv = DilConv(channel, channel, 3, 1, 2, 2, affine=True, upsample=True)
        # self.r1_l1_sge = sge_layer(16)
        # self.r1_l1_dilconvdouble = DilConvDouble(channel, channel, 3, 1, 2, 2, affine=True, upsample=True)
        # self.r1_l1_eca = eca_layer(3)
        # self.r1_l1_dil4Conv = DilConv(channel, channel, 3, 1, 4, 4, affine=True, upsample=False)
        # self.r1_l1_conv = Conv(channel, channel, 3, 1, 1, affine=True, upsample=False)
        # self.r1_l1_da = DoubleAttentionLayer(channel, channel, channel)
        # self.r1_l1_se = SELayer(channel)
        self.r1_l1_sa = sa_layer(channel, groups=16)

        # self.r1_l13_zero = Zero(1, True)
        # self.r1_l13_skipconnect = Identity(False)
        # self.r1_l13_sepconv = SepConv(channel, channel, 3, 1, 1, affine=True, upsample=True)
        # self.r1_l13_sepconvdouble = SepConvDouble(channel, channel, 3, 1, 1, affine=True, upsample=True)
        # self.r1_l13_dilconv = DilConv(channel, channel, 3, 1, 2, 2, affine=True, upsample=True)
        # self.r1_l13_sge = sge_layer(16)
        # self.r1_l13_dilconvdouble = DilConvDouble(channel, channel, 3, 1, 2, 2, affine=True, upsample=True)
        # self.r1_l13_eca = eca_layer(3)
        # self.r1_l13_dil4Conv = DilConv(channel, channel, 3, 1, 4, 4, affine=True, upsample=False)
        # self.r1_l13_conv = Conv(channel, channel, 3, 1, 1, affine=True, upsample=False)
        # self.r1_l13_da = DoubleAttentionLayer(channel, channel, channel)
        self.r1_l13_se = SELayer(channel)
        # self.r1_l13_sa = sa_layer(channel, groups=16)

        # self.r1_l2_zero = Zero(1, True)
        # self.r1_l2_skipconnect = Identity(False)
        # self.r1_l2_sepconv = SepConv(channel, channel, 3, 1, 1, affine=True, upsample=True)
        # self.r1_l2_sepconvdouble = SepConvDouble(channel, channel, 3, 1, 1, affine=True, upsample=True)
        # self.r1_l2_dilconv = DilConv(channel, channel, 3, 1, 2, 2, affine=True, upsample=True)
        # self.r1_l2_sge = sge_layer(16)
        # self.r1_l2_dilconvdouble = DilConvDouble(channel, channel, 3, 1, 2, 2, affine=True, upsample=True)
        # self.r1_l2_eca = eca_layer(3)
        self.r1_l2_dil4Conv = DilConv(channel, channel, 3, 1, 4, 4, affine=True, upsample=False)
        # self.r1_l2_conv = Conv(channel, channel, 3, 1, 1, affine=True, upsample=False)
        # self.r1_l2_da = DoubleAttentionLayer(channel, channel, channel)
        # self.r1_l2_se = SELayer(channel)
        # self.r1_l2_sa = sa_layer(channel, groups=16)

        # self.r1_l3_zero = Zero(1, False)
        # self.r1_l3_skipconnect = Identity(False)
        # self.r1_l3_sepconv = SepConv(channel, channel, 3, 1, 1, affine=True, upsample=False)
        # self.r1_l3_sepconvdouble = SepConvDouble(channel, channel, 3, 1, 1, affine=True, upsample=False)
        # self.r1_l3_dilconv = DilConv(channel, channel, 3, 1, 2, 2, affine=True, upsample=False)
        # self.r1_l3_sge = sge_layer(16)
        # self.r1_l3_dilconvdouble = DilConvDouble(channel, channel, 3, 1, 2, 2, affine=True, upsample=False)
        # self.r1_l3_eca = eca_layer(3)
        # self.r1_l3_dil4Conv = DilConv(channel, channel, 3, 1, 4, 4, affine=True, upsample=False)
        # self.r1_l3_conv = Conv(channel, channel, 3, 1, 1, affine=True, upsample=False)
        self.r1_l3_da = DoubleAttentionLayer(channel, channel, channel)
        # self.r1_l3_se = SELayer(channel)
        # self.r1_l3_sa = sa_layer(channel, groups=16)

    def forward(self, x, y, z):
        xy = self.r1_l1_sa(x)
        # else:
        #     x = self.r1_l1_ca(x)
        xy_temp = xy
        if xy.size()[2:] != y.size()[2:]:
            xy = F.interpolate(xy, y.size()[2:], mode='bilinear')

        y = y + xy
        yz = self.r1_l2_dil4Conv(y)
        #     y = self.r1_l2_ca(y)
        yz_temp = yz
        if yz.size()[2:] != z.size()[2:]:
            yz = F.interpolate(yz, z.size()[2:], mode='bilinear')
        #
        # if select[3] == 0:
        #     xz = self.r1_l13_zero(x)
        xz = self.r1_l13_se(yz)

        if xz.size()[2:] != z.size()[2:]:
            xz = F.interpolate(xz, z.size()[2:], mode='bilinear')

        z = z + yz + xz

        z = self.r1_l3_da(z)

        z_temp = z
        fuse = z + xz + yz

        return fuse, xy_temp, yz_temp, z_temp

class Round1_chosen_dark(nn.Module):
    def __init__(self, channel=128):
        super(Round1_chosen_dark, self).__init__()
        # self.r1_l1_zero = Zero(1, True)
        # self.r1_l1_skipconnect = Identity(False)
        # self.r1_l1_sepconv = SepConv(channel, channel, 3, 1, 1, affine=True, upsample=True)
        # self.r1_l1_sepconvdouble = SepConvDouble(channel, channel, 3, 1, 1, affine=True, upsample=True)
        # self.r1_l1_dilconv = DilConv(channel, channel, 3, 1, 2, 2, affine=True, upsample=True)
        # self.r1_l1_sge = sge_layer(16)
        # self.r1_l1_dilconvdouble = DilConvDouble(channel, channel, 3, 1, 2, 2, affine=True, upsample=True)
        # self.r1_l1_eca = eca_layer(3)
        # self.r1_l1_dil4Conv = DilConv(channel, channel, 3, 1, 4, 4, affine=True, upsample=False)
        # self.r1_l1_conv = Conv(channel, channel, 3, 1, 1, affine=True, upsample=False)
        # self.r1_l1_da = DoubleAttentionLayer(channel, channel, channel)
        # self.r1_l1_se = SELayer(channel)
        self.r1_l1_sa = sa_layer(channel, groups=16)

        # self.r1_l13_zero = Zero(1, True)
        # self.r1_l13_skipconnect = Identity(False)
        # self.r1_l13_sepconv = SepConv(channel, channel, 3, 1, 1, affine=True, upsample=True)
        # self.r1_l13_sepconvdouble = SepConvDouble(channel, channel, 3, 1, 1, affine=True, upsample=True)
        # self.r1_l13_dilconv = DilConv(channel, channel, 3, 1, 2, 2, affine=True, upsample=True)
        # self.r1_l13_sge = sge_layer(16)
        # self.r1_l13_dilconvdouble = DilConvDouble(channel, channel, 3, 1, 2, 2, affine=True, upsample=True)
        # self.r1_l13_eca = eca_layer(3)
        # self.r1_l13_dil4Conv = DilConv(channel, channel, 3, 1, 4, 4, affine=True, upsample=False)
        # self.r1_l13_conv = Conv(channel, channel, 3, 1, 1, affine=True, upsample=False)
        self.r1_l13_da = DoubleAttentionLayer(channel, channel, channel)
        # self.r1_l13_se = SELayer(channel)
        # self.r1_l13_sa = sa_layer(channel, groups=16)

        # self.r1_l2_zero = Zero(1, True)
        # self.r1_l2_skipconnect = Identity(False)
        # self.r1_l2_sepconv = SepConv(channel, channel, 3, 1, 1, affine=True, upsample=True)
        # self.r1_l2_sepconvdouble = SepConvDouble(channel, channel, 3, 1, 1, affine=True, upsample=True)
        # self.r1_l2_dilconv = DilConv(channel, channel, 3, 1, 2, 2, affine=True, upsample=True)
        # self.r1_l2_sge = sge_layer(16)
        # self.r1_l2_dilconvdouble = DilConvDouble(channel, channel, 3, 1, 2, 2, affine=True, upsample=True)
        # self.r1_l2_eca = eca_layer(3)
        # self.r1_l2_dil4Conv = DilConv(channel, channel, 3, 1, 4, 4, affine=True, upsample=False)
        # self.r1_l2_conv = Conv(channel, channel, 3, 1, 1, affine=True, upsample=False)
        # self.r1_l2_da = DoubleAttentionLayer(channel, channel, channel)
        self.r1_l2_se = SELayer(channel)
        # self.r1_l2_sa = sa_layer(channel, groups=16)

        # self.r1_l3_zero = Zero(1, False)
        # self.r1_l3_skipconnect = Identity(False)
        # self.r1_l3_sepconv = SepConv(channel, channel, 3, 1, 1, affine=True, upsample=False)
        # self.r1_l3_sepconvdouble = SepConvDouble(channel, channel, 3, 1, 1, affine=True, upsample=False)
        # self.r1_l3_dilconv = DilConv(channel, channel, 3, 1, 2, 2, affine=True, upsample=False)
        self.r1_l3_sge = sge_layer(16)
        # self.r1_l3_dilconvdouble = DilConvDouble(channel, channel, 3, 1, 2, 2, affine=True, upsample=False)
        # self.r1_l3_eca = eca_layer(3)
        # self.r1_l3_dil4Conv = DilConv(channel, channel, 3, 1, 4, 4, affine=True, upsample=False)
        # self.r1_l3_conv = Conv(channel, channel, 3, 1, 1, affine=True, upsample=False)
        # self.r1_l3_da = DoubleAttentionLayer(channel, channel, channel)
        # self.r1_l3_se = SELayer(channel)
        # self.r1_l3_sa = sa_layer(channel, groups=16)

    def forward(self, x, y, z):
        xy = self.r1_l1_sa(x)
        # else:
        #     x = self.r1_l1_ca(x)
        xy_temp = xy
        if xy.size()[2:] != y.size()[2:]:
            xy = F.interpolate(xy, y.size()[2:], mode='bilinear')

        y = y + xy
        yz = self.r1_l2_se(y)
        #     y = self.r1_l2_ca(y)
        yz_temp = yz
        if yz.size()[2:] != z.size()[2:]:
            yz = F.interpolate(yz, z.size()[2:], mode='bilinear')
        #
        # if select[3] == 0:
        #     xz = self.r1_l13_zero(x)
        xz = self.r1_l13_da(yz)

        if xz.size()[2:] != z.size()[2:]:
            xz = F.interpolate(xz, z.size()[2:], mode='bilinear')

        z = z + yz + xz

        z = self.r1_l3_sge(z)

        z_temp = z
        fuse = z + xz + yz

        return fuse, xy_temp, yz_temp, z_temp

class Round2(nn.Module):
    def __init__(self):
        super(Round2, self).__init__()
        self.r2_l0_zero = Zero(1, True)
        self.r2_l0_skipconnect = Identity(True)
        self.r2_l0_sepconv = SepConv(128, 128, 3, 1, 1, affine=True, upsample=True)
        self.r2_l0_sepconvdouble = SepConvDouble(128, 128, 3, 1, 1, affine=True, upsample=True)
        self.r2_l0_dilconv = DilConv(128, 128, 3, 1, 2, 2, affine=True, upsample=True)
        self.r2_l0_dilconvdouble = DilConvDouble(128, 128, 3, 1, 2, 2, affine=True, upsample=True)
        self.r2_l0_dil4Conv = DilConv(128, 128, 3, 1, 4, 4, affine=True, upsample=True)
        self.r2_l0_conv = Conv(128, 128, 3, 1, 1, affine=True, upsample=True)
        self.r2_l0_convdouble = ConvDouble(128, 128, 3, 1, 1, affine=True, upsample=True)
        self.r2_l0_se = SELayer(128)

        self.r2_l1_zero = Zero(1, True)
        self.r2_l1_skipconnect = Identity(True)
        self.r2_l1_sepconv = SepConv(128, 128, 3, 1, 1, affine=True, upsample=True)
        self.r2_l1_sepconvdouble = SepConvDouble(128, 128, 3, 1, 1, affine=True, upsample=True)
        self.r2_l1_dilconv = DilConv(128, 128, 3, 1, 2, 2, affine=True, upsample=True)
        self.r2_l1_dilconvdouble = DilConvDouble(128, 128, 3, 1, 2, 2, affine=True, upsample=True)
        self.r2_l1_dil4Conv = DilConv(128, 128, 3, 1, 4, 4, affine=True, upsample=True)
        self.r2_l1_conv = Conv(128, 128, 3, 1, 1, affine=True, upsample=True)
        self.r2_l1_convdouble = ConvDouble(128, 128, 3, 1, 1, affine=True, upsample=True)
        self.r2_l1_se = SELayer(128)

        self.r2_l02_zero = Zero(1, True)
        self.r2_l02_skipconnect = Identity(True)
        self.r2_l02_sepconv = SepConv(128, 128, 3, 1, 1, affine=True, upsample=True)
        self.r2_l02_sepconvdouble = SepConvDouble(128, 128, 3, 1, 1, affine=True, upsample=True)
        self.r2_l02_dilconv = DilConv(128, 128, 3, 1, 2, 2, affine=True, upsample=True)
        self.r2_l02_dilconvdouble = DilConvDouble(128, 128, 3, 1, 2, 2, affine=True, upsample=True)
        self.r2_l02_dil4Conv = DilConv(128, 128, 3, 1, 4, 4, affine=True, upsample=True)
        self.r2_l02_conv = Conv(128, 128, 3, 1, 1, affine=True, upsample=True)
        self.r2_l02_convdouble = ConvDouble(128, 128, 3, 1, 1, affine=True, upsample=True)
        self.r2_l02_se = SELayer(128)

        self.r2_l2_zero = Zero(1, True)
        self.r2_l2_skipconnect = Identity(True)
        self.r2_l2_sepconv = SepConv(128, 128, 3, 1, 1, affine=True, upsample=True)
        self.r2_l2_sepconvdouble = SepConvDouble(128, 128, 3, 1, 1, affine=True, upsample=True)
        self.r2_l2_dilconv = DilConv(128, 128, 3, 1, 2, 2, affine=True, upsample=True)
        self.r2_l2_dilconvdouble = DilConvDouble(128, 128, 3, 1, 2, 2, affine=True, upsample=True)
        self.r2_l2_dil4Conv = DilConv(128, 128, 3, 1, 4, 4, affine=True, upsample=True)
        self.r2_l2_conv = Conv(128, 128, 3, 1, 1, affine=True, upsample=True)
        self.r2_l2_convdouble = ConvDouble(128, 128, 3, 1, 1, affine=True, upsample=True)
        self.r2_l2_se = SELayer(128)

        self.r2_l3_zero = Zero(1, False)
        self.r2_l3_skipconnect = Identity(False)
        self.r2_l3_sepconv = SepConv(128, 128, 3, 1, 1, affine=True, upsample=False)
        self.r2_l3_sepconvdouble = SepConvDouble(128, 128, 3, 1, 1, affine=True, upsample=False)
        self.r2_l3_dilconv = DilConv(128, 128, 3, 1, 2, 2, affine=True, upsample=False)
        self.r2_l3_dilconvdouble = DilConvDouble(128, 128, 3, 1, 2, 2, affine=True, upsample=False)
        self.r2_l3_dil4Conv = DilConv(128, 128, 3, 1, 4, 4, affine=True, upsample=False)
        self.r2_l3_conv = Conv(128, 128, 3, 1, 1, affine=True, upsample=False)
        self.r2_l3_convdouble = ConvDouble(128, 128, 3, 1, 1, affine=True, upsample=False)
        self.r2_l3_se = SELayer(128)

        self.r2_l03_zero = Zero(1, False)
        self.r2_l03_skipconnect = Identity(False)
        self.r2_l03_sepconv = SepConv(128, 128, 3, 1, 1, affine=True, upsample=False)
        self.r2_l03_sepconvdouble = SepConvDouble(128, 128, 3, 1, 1, affine=True, upsample=False)
        self.r2_l03_dilconv = DilConv(128, 128, 3, 1, 2, 2, affine=True, upsample=False)
        self.r2_l03_dilconvdouble = DilConvDouble(128, 128, 3, 1, 2, 2, affine=True, upsample=False)
        self.r2_l03_dil4Conv = DilConv(128, 128, 3, 1, 4, 4, affine=True, upsample=False)
        self.r2_l03_conv = Conv(128, 128, 3, 1, 1, affine=True, upsample=False)
        self.r2_l03_convdouble = ConvDouble(128, 128, 3, 1, 1, affine=True, upsample=False)
        self.r2_l03_se = SELayer(128)

    def forward(self, fuse, x, y, z, select):
        if fuse.size()[2:] != x.size()[2:]:
            fusex = F.interpolate(fuse, x.size()[2:], mode='bilinear')

        if select[0] == 0:
            x0 = self.r2_l0_zero(fusex)
        elif select[0] == 1:
            x0 = self.r2_l0_skipconnect(fusex)
        elif select[0] == 2:
            x0 = self.r2_l0_sepconv(fusex)
        elif select[0] == 3:
            x0 = self.r2_l0_sepconvdouble(fusex)
        elif select[0] == 4:
            x0 = self.r2_l0_dilconv(fusex)
        elif select[0] == 5:
            x0 = self.r2_l0_dilconvdouble(fusex)
        elif select[0] == 6:
            x0 = self.r2_l0_dil4Conv(fusex)
        elif select[0] == 7:
            x0 = self.r2_l0_conv(fusex)
        elif select[0] == 8:
            x0 = self.r2_l0_convdouble(fusex)
        elif select[0] == 9:
            x0 = self.r2_l0_se(fusex)
        # else:
        #     x = self.r1_l1_ca(x)
        if x0.size()[2:] != x.size()[2:]:
            x0 = F.interpolate(x0, x.size()[2:], mode='bilinear')

        x = x + x0
        if select[1] == 0:
            xy = self.r2_l1_zero(x)
        elif select[1] == 1:
            xy = self.r2_l1_skipconnect(y)
        elif select[1] == 2:
            xy = self.r2_l1_sepconv(y)
        elif select[1] == 3:
            xy = self.r2_l1_sepconvdouble(y)
        elif select[1] == 4:
            xy = self.r2_l1_dilconv(y)
        elif select[1] == 5:
            xy = self.r2_l1_dilconvdouble(y)
        elif select[1] == 6:
            xy = self.r2_l1_dil4Conv(y)
        elif select[1] == 7:
            xy = self.r2_l1_conv(y)
        elif select[1] == 8:
            xy = self.r2_l1_convdouble(y)
        elif select[1] == 9:
            xy = self.r2_l1_se(y)

        if xy.size()[2:] != y.size()[2:]:
            xy = F.interpolate(xy, y.size()[2:], mode='bilinear')

        ###
        if select[2] == 0:
            x0y = self.r2_l02_zero(x0)
        elif select[2] == 1:
            x0y = self.r2_l02_skipconnect(x0)
        elif select[2] == 2:
            x0y = self.r2_l02_sepconv(x0)
        elif select[2] == 3:
            x0y = self.r2_l02_sepconvdouble(x0)
        elif select[2] == 4:
            x0y = self.r2_l02_dilconv(x0)
        elif select[2] == 5:
            x0y = self.r2_l02_dilconvdouble(x0)
        elif select[2] == 6:
            x0y = self.r2_l02_dil4Conv(x0)
        elif select[2] == 7:
            x0y = self.r2_l02_conv(x0)
        elif select[2] == 8:
            x0y = self.r2_l02_convdouble(x0)
        elif select[2] == 9:
            x0y = self.r2_l02_se(x0)

        if x0y.size()[2:] != y.size()[2:]:
            x0y = F.interpolate(x0y, y.size()[2:], mode='bilinear')

        y = y + xy + x0y
        if select[3] == 0:
            yz = self.r2_l2_zero(y)
        elif select[3] == 1:
            yz = self.r2_l2_skipconnect(y)
        elif select[3] == 2:
            yz = self.r2_l2_sepconv(y)
        elif select[3] == 3:
            yz = self.r2_l2_sepconvdouble(y)
        elif select[3] == 4:
            yz = self.r2_l2_dilconv(y)
        elif select[3] == 5:
            yz = self.r2_l2_dilconvdouble(y)
        elif select[3] == 6:
            yz = self.r2_l2_dil4Conv(y)
        elif select[3] == 7:
            yz = self.r2_l2_conv(y)
        elif select[3] == 8:
            yz = self.r2_l2_convdouble(y)
        elif select[3] == 9:
            yz = self.r2_l2_se(y)

        if yz.size()[2:] != z.size()[2:]:
            yz = F.interpolate(yz, z.size()[2:], mode='bilinear')

        if select[4] == 0:
            x0z = self.r2_l03_zero(fusex)
        elif select[4] == 1:
            x0z = self.r2_l03_skipconnect(fusex)
        elif select[4] == 2:
            x0z = self.r2_l03_sepconv(fusex)
        elif select[4] == 3:
            x0z = self.r2_l03_sepconvdouble(fusex)
        elif select[4] == 4:
            x0z = self.r2_l03_dilconv(fusex)
        elif select[4] == 5:
            x0z = self.r2_l03_dilconvdouble(fusex)
        elif select[4] == 6:
            x0z = self.r2_l03_dil4Conv(fusex)
        elif select[4] == 7:
            x0z = self.r2_l03_conv(fusex)
        elif select[4] == 8:
            x0z = self.r2_l03_convdouble(fusex)
        elif select[4] == 9:
            x0z = self.r2_l03_se(x)
        # else:
        #     x = self.r1_l1_ca(x)
        if x0z.size()[2:] != z.size()[2:]:
            x0z = F.interpolate(x0z, z.size()[2:], mode='bilinear')

        z = z + yz + x0z

        if select[5] == 0:
            z = self.r2_l3_zero(z)
        elif select[5] == 1:
            z = self.r2_l3_skipconnect(z)
        elif select[5] == 2:
            z = self.r2_l3_sepconv(z)
        elif select[5] == 3:
            z = self.r2_l3_sepconvdouble(z)
        elif select[5] == 4:
            z = self.r2_l3_dilconv(z)
        elif select[5] == 5:
            z = self.r2_l3_dilconvdouble(z)
        elif select[5] == 6:
            z = self.r2_l3_dil4Conv(z)
        elif select[5] == 7:
            z = self.r2_l3_conv(z)
        elif select[5] == 8:
            z = self.r2_l3_convdouble(z)
        elif select[5] == 9:
            z = self.r2_l3_se(z)
        # else:
        #     z = self.r1_l3_ca(z)
        if xy.size()[2:] != z.size()[2:]:
            xy = F.interpolate(xy, z.size()[2:], mode='bilinear')
        if yz.size()[2:] != z.size()[2:]:
            yz = F.interpolate(yz, z.size()[2:], mode='bilinear')

        fuse2 = z + xy + yz

        return fuse2

class Search(nn.Module):
    def __init__(self, channel=128):
        super(Search, self).__init__()
        # round 1(1~3), level 1(1~3), op:1~12
        self.round = Round1_chosen_dark(channel=channel)
        # self.round2 = Round1()

    def forward(self, x, y, z, select):
        # print(select)
        feedback = self.round(x, y, z)
        # fb_x = F.interpolate(feedback, x.size()[2:], mode='bilinear')
        # fb_y = F.interpolate(feedback, y.size()[2:], mode='bilinear')
        # feedback2 = self.round2(x + fb_x, y + fb_y, z + feedback, select[4:])
        return feedback

if __name__ == '__main__':
    x = torch.zeros(2, 128, 32, 32)
    y = torch.zeros(2, 128, 64, 64)
    z = torch.zeros(2, 128, 128, 128)

    model = Search()
    choice = [9, 9, 9, 9, 9, 9, 9, 9, 0, 0]
    output, _ = model(x, y, z, choice)
    print(_.size())