import torch
import torch.nn as nn
import torch.nn.functional as F
# from underwater_model.base import Base
from underwater_model.base_SPOS import Base
from underwater_model.search_net import Search
from underwater_model.color_SPOS import Color
from underwater_model.vit import ViT

class Water(nn.Module):
    def __init__(self, en_channels, de_channels):
        super(Water, self).__init__()

        self.base = Base(en_channels)

        self.align1 = nn.Sequential(nn.Conv2d(en_channels[0] * 2, de_channels, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=True))

        self.align2 = nn.Sequential(nn.Conv2d(en_channels[1] * 2, de_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(inplace=True))

        self.align3 = nn.Sequential(nn.Conv2d(en_channels[2] * 2, de_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(inplace=True))
        # vit
        # self.vit1 = ViT(image_size=128, patch_size=16, dim=128, depth=1, heads=1, mlp_dim=128, channels=128)
        # self.vit2 = ViT(image_size=64, patch_size=8, dim=128, depth=1, heads=1, mlp_dim=128, channels=128)
        # self.vit3 = ViT(image_size=32, patch_size=4, dim=128, depth=1, heads=1, mlp_dim=128, channels=128)

        self.search = Search(channel=de_channels)
        # self.search2 = Search(channel=de_channels)
        # self.color = Color(channel=de_channels)

        # self.de_predict_color = nn.Sequential(nn.Conv2d(de_channels, 2, kernel_size=1, stride=1))

        self.de_predict = nn.Sequential(nn.Conv2d(de_channels, 2, kernel_size=1, stride=1))
        self.de_predict2 = nn.Sequential(nn.Conv2d(de_channels, 3, kernel_size=1, stride=1))
        # self.de_predict_conv1_ab = nn.Sequential(nn.Conv2d(de_channels, 128, kernel_size=1, stride=1), nn.ReLU(inplace=True))
        # self.de_predict_conv2_ab = nn.Sequential(nn.Conv2d(de_channels, 2, kernel_size=1, stride=1))
        # self.de_predict_lab_final = nn.Sequential(nn.Conv2d(de_channels, 1, kernel_size=1, stride=1))
        # self.de_predict2 = nn.Sequential(nn.Conv2d(128, 3, kernel_size=1, stride=1))
        self.de_predict_rgb = nn.Sequential(nn.Conv2d(en_channels[2], 3, kernel_size=1, stride=1))
        # self.de_predict_hsv = nn.Sequential(nn.Conv2d(en_channels[2], 3, kernel_size=1, stride=1))
        self.de_predict_lab = nn.Sequential(nn.Conv2d(en_channels[2], 3, kernel_size=1, stride=1))
    def forward(self, rgb, hsv, lab, trans_map, select):
        # trans_map2 = F.max_pool2d(1 - trans_map, kernel_size=3, stride=2, padding=1)
        # trans_map3 = F.max_pool2d(trans_map2, kernel_size=3, stride=2, padding=1)
        # x_lab3 = F.max_pool2d(x_lab3, kernel_size=3, stride=2, padding=1)
        # first_rgb, first_hsv, first_lab, second_rgb, second_hsv, second_lab, third_rgb, \
        # third_hsv, third_lab = self.base(rgb, hsv, lab, select[:18])
        first_rgb, first_lab, second_rgb, second_lab, third_rgb, \
        third_lab = self.base(rgb, hsv, lab, select[:12])

        # first_rgb, second_rgb, third_rgb = self.base(rgb, hsv, lab, select[:6])

        # first_rgb, first_hsv, first_lab, second_rgb, second_hsv, second_lab, third_rgb, \
        # third_hsv, third_lab = self.base(rgb, hsv, lab)
        # first = first + first * (1 - trans_map)
        inter_rgb = F.interpolate(self.de_predict_rgb(third_rgb), rgb.size()[2:], mode='bilinear')
        # inter_hsv = F.interpolate(self.de_predict_hsv(third_hsv), hsv.size()[2:], mode='bilinear')
        inter_lab = F.interpolate(self.de_predict_lab(third_lab), lab.size()[2:], mode='bilinear')

        # if select[0] == 0:
        first = self.align1(torch.cat([first_rgb, first_lab], dim=1))

        second = self.align2(torch.cat([second_rgb, second_lab], dim=1))

        third = self.align3(torch.cat([third_rgb, third_lab], dim=1))
        # first = self.align1(first_rgb)
        #
        # second = self.align2(second_rgb)
        #
        # third = self.align3(third_rgb)


        mid_ab_feat, _, _, _ = self.search(third, second, first, select[12:16])
        # final_color = self.search_color(third, second, first, select[16:])
        final_ab = self.de_predict(mid_ab_feat)
        # mid_ab_feat = self.de_predict_conv1_ab(final_feat)
        # mid_ab = self.de_predict_conv2_ab(mid_ab_feat)
        mid_ab_feat3 = F.interpolate(mid_ab_feat, size=third.size()[2:], mode='bilinear')
        mid_ab_feat2 = F.interpolate(mid_ab_feat, size=second.size()[2:], mode='bilinear')
        final2, third, second, first = self.search(third + mid_ab_feat3,
                                                   second + mid_ab_feat2, first + mid_ab_feat, select[16:])
        # temp_gray = torch.mean(final_rgb, dim=1, keepdim=True)
        # final_lab = self.de_predict_color_final(final_color)
        # final_lab = torch.cat([temp_gray, final_lab], dim=1)
        # final2 = self.de_predict2(final2)
        final2_rgb = self.de_predict2(final2)

        return final_ab, final2_rgb, inter_rgb, inter_lab

if __name__ == '__main__':
    a = torch.zeros(2, 3, 128, 128)
    b = torch.zeros(2, 1, 128, 128)

    model = Water(en_channels=[64, 128, 256], de_channels=128)
    r = model(a, a, a, b, [1, 7, 6, 5, 4, 5, 5, 1, 3, 5, 5, 6, 6, 4, 6, 3, 3, 6, 2, 1])
    print(r[0].shape)



