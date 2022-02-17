import torch
import torch.nn as nn
import torch.nn.functional as F
# from underwater_model.base import Base
from underwater_model.base_SPOS import Base
from underwater_model.search_net import Search
from underwater_model.color_SPOS import Color
from underwater_model.vit import ViT
from underwater_model.refine import refine_block

class RCTConvBlock(nn.Module):
    def __init__(self, input_nc, output_nc, ksize=3, stride=2, pad=1, extra_conv=False):
        super(RCTConvBlock, self).__init__()

        lists = []

        if extra_conv:
            lists += [
                nn.Conv2d(input_nc, input_nc, kernel_size=(ksize, ksize), stride=(stride, stride), padding=(pad, pad)),
                nn.BatchNorm2d(input_nc),
                nn.SiLU(inplace=True),  # Swish activation
                nn.Conv2d(input_nc, output_nc, kernel_size=(ksize, ksize), stride=(stride, stride), padding=(pad, pad))
            ]
        else:
            lists += [
                nn.Conv2d(input_nc, output_nc, kernel_size=(ksize, ksize), stride=(stride, stride), padding=(pad, pad)),
                nn.BatchNorm2d(output_nc),
                nn.SiLU(inplace=True)
            ]

        self.model = nn.Sequential(*lists)

    def forward(self, x):
        return self.model(x)

def pad_tensor(input, divide):
    height_org, width_org = input.shape[2], input.shape[3]

    if width_org % divide != 0 or height_org % divide != 0:

        width_res = width_org % divide
        height_res = height_org % divide
        if width_res != 0:
            width_div = divide - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0

        if height_res != 0:
            height_div = divide - height_res
            pad_top = int(height_div / 2)
            pad_bottom = int(height_div - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0

        padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
        input = padding(input).data
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    height, width = input.shape[2], input.shape[3]
    assert width % divide == 0, 'width cant divided by stride'
    assert height % divide == 0, 'height cant divided by stride'

    return input, pad_left, pad_right, pad_top, pad_bottom


def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
    height, width = input.shape[2], input.shape[3]
    return input[:, :, pad_top: height - pad_bottom, pad_left: width - pad_right]

class GlobalRCT(nn.Module):
    def __init__(self, fusion_filter, represent_feature, ngf):
        super(GlobalRCT, self).__init__()
        self.represent_feature = represent_feature
        self.ngf = ngf
        self.r_conv = RCTConvBlock(fusion_filter, represent_feature * ngf, 1, 1, 0, True)
        self.t_conv = RCTConvBlock(fusion_filter, 3 * ngf, 1, 1, 0, True)
        self.act = nn.Softmax(dim=2)

    def forward(self, feature, p_high):
        h, w = feature.shape[2], feature.shape[3]
        f_r = feature.reshape(feature.size(0), self.represent_feature, -1)
        f_r = f_r.transpose(1, 2)
        r_g = self.r_conv(p_high)
        r_g = r_g.reshape(r_g.size(0), self.represent_feature, self.ngf)
        t_g = self.t_conv(p_high)
        t_g = t_g.reshape(t_g.size(0), 3, self.ngf)

        attention = torch.bmm(f_r, r_g) / torch.sqrt(torch.tensor(self.represent_feature))
        attention = self.act(attention)
        Y_G = torch.bmm(attention, t_g.transpose(1, 2))
        Y_G = Y_G.transpose(1, 2)
        return Y_G.reshape(Y_G.size(0), 3, h, w)

class LocalRCT(nn.Module):
    def __init__(self, fusion_filter, represent_feature, nlf, mesh_size):
        super(LocalRCT, self).__init__()

        self.fusion_filter = fusion_filter
        self.represent_feature = represent_feature
        self.nlf = nlf
        self.mesh_size = mesh_size

        self.r_conv = RCTConvBlock(self.fusion_filter, self.represent_feature * self.nlf, 3, 1, 1, True)
        self.t_conv = RCTConvBlock(self.fusion_filter, 3 * self.nlf, 3, 1, 1, True)
        self.act = nn.Softmax(dim=2)


    def forward(self, feature, p_low):
        nfeature, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(feature, self.mesh_size)
        mesh_h = int(nfeature.shape[2] / self.mesh_size)
        mesh_w = int(nfeature.shape[3] / self.mesh_size)

        r_l = self.r_conv(p_low)
        t_l = self.t_conv(p_low)
        Y_L = torch.zeros(nfeature.size(0), 3, nfeature.size(2), nfeature.size(3), device=feature.device)

        # Grid-wise
        for i in range(self.mesh_size):
            for j in range(self.mesh_size):
                # cp means corner points of a grid
                r_k = r_l[:, :, i, j].reshape(-1, self.represent_feature, self.nlf)
                cp = r_l[:, :, i, j + 1].reshape(-1, self.represent_feature, self.nlf)
                r_k = torch.cat((r_k, cp), dim=2)
                cp = r_l[:, :, i + 1, j].reshape(-1, self.represent_feature, self.nlf)
                r_k = torch.cat((r_k, cp), dim=2)
                cp = r_l[:, :, i + 1, j + 1].reshape(-1, self.represent_feature, self.nlf)
                r_k = torch.cat((r_k, cp), dim=2)

                t_k = t_l[:, :, i, j].reshape(-1, 3, self.nlf)
                cp = t_l[:, :, i, j + 1].reshape(-1, 3, self.nlf)
                t_k = torch.cat((t_k, cp), dim=2)
                cp = t_l[:, :, i + 1, j].reshape(-1, 3, self.nlf)
                t_k = torch.cat((t_k, cp), dim=2)
                cp = t_l[:, :, i + 1, j + 1].reshape(-1, 3, self.nlf)
                t_k = torch.cat((t_k, cp), dim=2)

                f_k = nfeature[:, :, i * mesh_h:(i + 1) * mesh_h, j * mesh_w:(j + 1) * mesh_w]
                f_k = f_k.reshape(feature.size(0), self.represent_feature, -1)
                f_k = f_k.transpose(1, 2)

                attention = torch.bmm(f_k, r_k) / torch.sqrt(torch.tensor(self.represent_feature))
                attention = self.act(attention)
                mesh = torch.bmm(attention, t_k.transpose(1, 2))
                mesh = mesh.transpose(1, 2)
                Y_L[:, :, i * mesh_h:(i + 1) * mesh_h, j * mesh_w:(j + 1) * mesh_w] = mesh.reshape(mesh.size(0), 3,
                                                                                                   mesh_h, mesh_w)

        return pad_tensor_back(Y_L, pad_left, pad_right, pad_top, pad_bottom)

class Water(nn.Module):
    def __init__(self, en_channels, de_channels):
        super(Water, self).__init__()

        self.base = Base(en_channels)

        self.align1 = nn.Sequential(nn.Conv2d(en_channels[0] * 1, de_channels, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=True))

        self.align2 = nn.Sequential(nn.Conv2d(en_channels[1] * 1, de_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(inplace=True))

        self.align3 = nn.Sequential(nn.Conv2d(en_channels[2] * 1, de_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(inplace=True))
        # vit
        # self.vit1 = ViT(image_size=128, patch_size=16, dim=128, depth=1, heads=1, mlp_dim=128, channels=128)
        # self.vit2 = ViT(image_size=64, patch_size=8, dim=128, depth=1, heads=1, mlp_dim=128, channels=128)
        # self.vit3 = ViT(image_size=56, patch_size=7, dim=128, depth=1, heads=1, mlp_dim=128, channels=128)
        self.refine = refine_block(128, 64)

        self.search = Search(channel=de_channels)
        # self.search2 = Search(channel=de_channels)
        # self.color = Color(channel=de_channels)

        # self.de_predict_color = nn.Sequential(nn.Conv2d(de_channels, 2, kernel_size=1, stride=1))

        self.de_predict = nn.Sequential(nn.Conv2d(de_channels, 3, kernel_size=1, stride=1))
        # self.de_predict2 = nn.Sequential(nn.Conv2d(de_channels, 3, kernel_size=1, stride=1))

        self.de_predict_rgb = nn.Sequential(nn.Conv2d(en_channels[2], 3, kernel_size=1, stride=1))

        self.de_predict_lab = nn.Sequential(nn.Conv2d(en_channels[2], 3, kernel_size=1, stride=1))

    def forward(self, rgb, lab, select):

        # first_rgb, first_lab, second_rgb, second_lab, third_rgb, \
        # third_lab = self.base(rgb, lab, select[:6])


        first_rgb, second_rgb, third_rgb = self.base(rgb, lab, select[:6])



        inter_rgb = F.interpolate(self.de_predict_rgb(third_rgb), rgb.size()[2:], mode='bilinear')
        # inter_lab = F.interpolate(self.de_predict_lab(third_lab), lab.size()[2:], mode='bilinear')


        # first = self.align1(torch.cat([first_rgb, first_lab], dim=1))
        first = self.align1(first_rgb)
        # second = self.align2(torch.cat([second_rgb, second_lab], dim=1))
        second = self.align2(second_rgb)
        # third = self.align3(torch.cat([third_rgb, third_lab], dim=1))
        third = self.align3(third_rgb)


        third = self.refine(third, select[6])
        mid_ab_feat, _, _, _ = self.search(third, second, first, select[7:])

        final_ab = self.de_predict(mid_ab_feat)

        # mid_ab_feat3 = F.interpolate(mid_ab_feat, size=third.size()[2:], mode='bilinear')
        # mid_ab_feat2 = F.interpolate(mid_ab_feat, size=second.size()[2:], mode='bilinear')
        # final2, third, second, first = self.search(third + mid_ab_feat3,
        #                                            second + mid_ab_feat2, first + mid_ab_feat, select[17:])
        # final2 = self.de_predict2(final2)
        # p = F.adaptive_avg_pool2d(mid_ab_feat, 1)
        # p16 = F.adaptive_avg_pool2d(mid_ab_feat, 16)
        # final2_rgb_rct = self.global_rct(final2, p)
        # final2_rgb_rct2 = self.local_rct(final2, p16)

        # final2 = F.relu(self.w[0]) * self.global_rct(final2, p) + F.relu(self.w[1]) * self.local_rct(final2, p16)
        # temp_gray = torch.mean(final_rgb, dim=1, keepdim=True)
        # final_lab = self.de_predict_color_final(final_color)
        # final_lab = torch.cat([temp_gray, final_lab], dim=1)
        # final2 = self.de_predict2(final2)
        # final2_rgb = self.de_predict2(final2)
        # third = self.de_predict_third(third)
        # second = self.de_predict_second(second)

        return final_ab, inter_rgb

if __name__ == '__main__':
    a = torch.zeros(2, 128, 200, 300)
    b = torch.zeros(2, 128, 1, 1)

    # model = Water(en_channels=[64, 128, 256], de_channels=128)
    # r, r1, r2, r3 = model(a, a, a, b, [1, 7, 6, 5, 4, 5, 5, 1, 3, 5, 5, 6, 6, 4, 6, 3, 3, 6, 2, 1])
    # print(r1.shape, '--', r2.shape, '--', r3.shape)

    c = torch.zeros(2, 128, 48, 64).cuda()

    # global_rct = GlobalRCT(128, 128, 8)

    local_rct = LocalRCT(128, 128, 8, 15).cuda()

    # out = global_rct(a, b)
    # global_rct = LocalRCT(128, 128, 8, 16)
    out = local_rct(a, c)
    print(out.shape)




