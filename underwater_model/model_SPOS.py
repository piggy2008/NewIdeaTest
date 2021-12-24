import torch
import torch.nn as nn
import torch.nn.functional as F
# from underwater_model.base import Base
from underwater_model.base_SPOS import Base
from underwater_model.search_net import Search

class Water(nn.Module):
    def __init__(self):
        super(Water, self).__init__()

        self.base = Base()

        self.align1 = nn.Sequential(nn.Conv2d(128 * 3, 128, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=True))

        self.align2 = nn.Sequential(nn.Conv2d(256 * 3, 128, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(inplace=True))

        self.align3 = nn.Sequential(nn.Conv2d(512 * 3, 128, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(inplace=True))
        #
        self.align1_rgb_hsv = nn.Sequential(nn.Conv2d(128 * 2, 128, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=True))

        self.align1_rgb_lab = nn.Sequential(nn.Conv2d(128 * 2, 128, kernel_size=3, stride=1, padding=1),
                                            nn.ReLU(inplace=True))

        self.align1_hsv_lab = nn.Sequential(nn.Conv2d(128 * 2, 128, kernel_size=3, stride=1, padding=1),
                                            nn.ReLU(inplace=True))

        self.align2_rgb_hsv = nn.Sequential(nn.Conv2d(256 * 2, 128, kernel_size=3, stride=1, padding=1),
                                            nn.ReLU(inplace=True))

        self.align2_rgb_lab = nn.Sequential(nn.Conv2d(256 * 2, 128, kernel_size=3, stride=1, padding=1),
                                            nn.ReLU(inplace=True))

        self.align2_hsv_lab = nn.Sequential(nn.Conv2d(256 * 2, 128, kernel_size=3, stride=1, padding=1),
                                            nn.ReLU(inplace=True))

        self.align3_rgb_hsv = nn.Sequential(nn.Conv2d(512 * 2, 128, kernel_size=3, stride=1, padding=1),
                                            nn.ReLU(inplace=True))

        self.align3_rgb_lab = nn.Sequential(nn.Conv2d(512 * 2, 128, kernel_size=3, stride=1, padding=1),
                                            nn.ReLU(inplace=True))

        self.align3_hsv_lab = nn.Sequential(nn.Conv2d(512 * 2, 128, kernel_size=3, stride=1, padding=1),
                                            nn.ReLU(inplace=True))

        self.search = Search()

        self.de_predict = nn.Sequential(nn.Conv2d(128, 3, kernel_size=1, stride=1))
        # self.de_predict2 = nn.Sequential(nn.Conv2d(128, 3, kernel_size=1, stride=1))
        self.de_predict_rgb = nn.Sequential(nn.Conv2d(512, 3, kernel_size=1, stride=1))
        self.de_predict_hsv = nn.Sequential(nn.Conv2d(512, 3, kernel_size=1, stride=1))
        self.de_predict_lab = nn.Sequential(nn.Conv2d(512, 3, kernel_size=1, stride=1))
    def forward(self, rgb, hsv, lab, trans_map, select):
        trans_map2 = F.max_pool2d(1 - trans_map, kernel_size=3, stride=2, padding=1)
        trans_map3 = F.max_pool2d(trans_map2, kernel_size=3, stride=2, padding=1)
        # x_lab3 = F.max_pool2d(x_lab3, kernel_size=3, stride=2, padding=1)
        first_rgb, first_hsv, first_lab, second_rgb, second_hsv, second_lab, third_rgb, \
        third_hsv, third_lab = self.base(rgb, hsv, lab, select[:6])
        # first = first + first * (1 - trans_map)
        inter_rgb = F.interpolate(self.de_predict_rgb(third_rgb), rgb.size()[2:], mode='bilinear')
        inter_hsv = F.interpolate(self.de_predict_hsv(third_hsv), hsv.size()[2:], mode='bilinear')
        inter_lab = F.interpolate(self.de_predict_lab(third_lab), lab.size()[2:], mode='bilinear')

        # if select[0] == 0:
        first = self.align1(torch.cat([first_rgb, first_hsv, first_lab], dim=1))
        # elif select[0] == 1:
        #     first = self.align1_rgb_hsv(torch.cat([first_rgb, first_hsv], dim=1))
        # elif select[0] == 2:
        #     first = self.align1_rgb_lab(torch.cat([first_rgb, first_lab], dim=1))
        # elif select[0] == 3:
        #     first = self.align1_hsv_lab(torch.cat([first_hsv, first_lab], dim=1))
        # if select[0] in [0, 1, 2, 3, 4]:
        # second = second + second * trans_map2
        # if select[1] == 0:
        second = self.align2(torch.cat([second_rgb, second_hsv, second_lab], dim=1))
        # elif select[1] == 1:
        #     second = self.align2_rgb_hsv(torch.cat([second_rgb, second_hsv], dim=1))
        # elif select[1] == 2:
        #     second = self.align2_rgb_lab(torch.cat([second_rgb, second_lab], dim=1))
        # elif select[1] == 3:
        #     second = self.align2_hsv_lab(torch.cat([second_hsv, second_lab], dim=1))
        # if select[1] in [0, 1, 2, 3, 4]:

        # third = third + third * trans_map3
        # if select[2] == 0:
        third = self.align3(torch.cat([third_rgb, third_hsv, third_lab], dim=1))
        # elif select[2] == 1:
        #     third = self.align3_rgb_hsv(torch.cat([third_rgb, third_hsv], dim=1))
        # elif select[2] == 2:
        #     third = self.align3_rgb_lab(torch.cat([third_rgb, third_lab], dim=1))
        # elif select[2] == 3:
        #     third = self.align3_hsv_lab(torch.cat([third_hsv, third_lab], dim=1))
        # if select[2] in [0, 1, 2, 3, 4]:
        # print(select[:6])
        final = self.search(third, second, first, select[6:])
        final = self.de_predict(final)
        # final2 = self.de_predict2(final2)

        return final, inter_rgb, inter_hsv, inter_lab

if __name__ == '__main__':
    a = torch.zeros(2, 3, 128, 128)
    b = torch.zeros(2, 1, 128, 128)

    model = Water()
    r = model(a, a, a, b, [1, 2, 3, 1, 2, 3, 1, 2, 3, 1])
    print(r[0].shape)




