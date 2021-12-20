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

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        # self.conv_align = nn.Sequential(nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1),
        #                                 nn.ReLU(inplace=True))

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        temp = x * y.expand_as(x) + x
        return temp

class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim * 3, out_channels=in_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim * 3, out_channels=in_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim * 3, out_channels=in_dim, kernel_size=1)

    def forward(self, rgb, hsv, lab):

        batch, channel, height, width = rgb.size()

        combined = torch.cat([rgb, hsv, lab], dim=1)

        proj_query = self.query_conv(combined).view(batch, channel, -1)
        proj_key = self.key_conv(combined).view(batch, channel, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy = ((self.chanel_in) ** -.5) * energy
        attention = F.softmax(energy)
        proj_value = self.value_conv(combined).view(batch, channel, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(batch, channel, height, width)
        return out


class Water(nn.Module):
    def __init__(self):
        super(Water, self).__init__()
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

        # self.cam1 = SELayer(128 * 3)
        # self.cam2 = SELayer(256 * 3)
        # self.cam3 = SELayer(512 * 3)

        self.de_block1 = resblock(n_channels=512)
        self.de_block2 = resblock(n_channels=512)

        self.de_block3 = resblock(n_channels=256)
        self.de_block4 = resblock(n_channels=256)

        self.de_block5 = resblock(n_channels=128)
        self.de_block6 = resblock(n_channels=128)

        self.de_conv = nn.Sequential(nn.Conv2d(256 + 512, 256, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=True))

        self.de_conv2 = nn.Sequential(nn.Conv2d(256 + 128, 128, kernel_size=3, stride=1, padding=1),
                                     nn.ReLU(inplace=True))
        self.de_predict = nn.Sequential(nn.Conv2d(128, 3, kernel_size=1, stride=1))
    def forward(self, rgb, hsv, lab, trans_map):
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
        # x_lab3 = F.max_pool2d(x_lab3, kernel_size=3, stride=2, padding=1)

        # first = self.cam1(torch.cat([x_rgb, x_hsv, x_lab], dim=1))
        # second = self.cam2(torch.cat([x_rgb2, x_hsv2, x_lab2], dim=1))
        # third = self.cam3(torch.cat([x_rgb3, x_hsv3, x_lab3], dim=1))

        first = x_rgb + x_hsv + x_lab
        second = x_rgb2 + x_hsv2 + x_lab2
        third = x_rgb3 + x_hsv3 + x_lab3

        trans_map2 = F.max_pool2d(1 - trans_map, kernel_size=3, stride=2, padding=1)
        trans_map3 = F.max_pool2d(trans_map2, kernel_size=3, stride=2, padding=1)
        # trans_map4 = F.max_pool2d(trans_map3, kernel_size=3, stride=2, padding=1)

        de_input = third + third * trans_map3
        de_input = self.de_block1(de_input)
        de_input = self.de_block2(de_input)
        de_input = F.interpolate(de_input, size=second.shape[2:], mode='bilinear')

        de_input2 = second + second * trans_map2
        de_input2 = torch.cat([de_input2, de_input], dim=1)
        de_input2 = self.de_conv(de_input2)
        de_input2 = self.de_block3(de_input2)
        de_input2 = self.de_block4(de_input2)
        de_input2 = F.interpolate(de_input2, size=first.shape[2:], mode='bilinear')

        de_input3 = first + first * (1 - trans_map)
        de_input3 = torch.cat([de_input3, de_input2], dim=1)
        de_input3 = self.de_conv2(de_input3)
        de_input3 = self.de_block5(de_input3)
        de_input3 = self.de_block6(de_input3)
        final = self.de_predict(de_input3)

        return final

if __name__ == '__main__':
    a = torch.zeros(2, 3, 128, 128)
    b = torch.zeros(2, 1, 128, 128)

    model = Water()
    r = model(a, a, a, b)
    print(r.shape)




