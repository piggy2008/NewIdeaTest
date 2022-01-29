import torch
import torch.nn as nn


class DisGeneralConvBlock(nn.Module):
    """ General block in the discriminator  """

    def __init__(self, in_channels, out_channels, use_eql=True):
        """
        constructor of the class
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param use_eql: whether to use equalized learning rate
        """
        from torch.nn import AvgPool2d, LeakyReLU
        from torch.nn import Conv2d

        super().__init__()

        if use_eql:
            self.conv_1 = _equalized_conv2d(in_channels, in_channels, (3, 3),
                                            pad=1, bias=True)
            self.conv_2 = _equalized_conv2d(in_channels, out_channels, (3, 3),
                                            pad=1, bias=True)
        else:
            # convolutional modules
            self.conv_1 = Conv2d(in_channels, in_channels, (3, 3),
                                 padding=1, bias=True)
            self.conv_2 = Conv2d(in_channels, out_channels, (3, 3),
                                 padding=1, bias=True)

        self.downSampler = AvgPool2d(2)  # downsampler

        # leaky_relu:
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the module
        :param x: input
        :return: y => output
        """
        # define the computations
        y = self.lrelu(self.conv_1(x))
        y = self.lrelu(self.conv_2(y))
        y = self.downSampler(y)

        return y


class from_rgb(nn.Module):
    """
    The RGB image is transformed into a multi-channel feature map to be concatenated with
    the feature map with the same number of channels in the network
    把RGB图转换为多通道特征图，以便与网络中相同通道数的特征图拼接
    """

    def __init__(self, outchannels, use_eql=True):
        super(from_rgb, self).__init__()
        if use_eql:
            self.conv_1 = _equalized_conv2d(3, outchannels, (1, 1), bias=True)
        else:
            self.conv_1 = nn.Conv2d(3, outchannels, (1, 1), bias=True)
        # pixel_wise feature normalizer:
        self.pixNorm = PixelwiseNorm()

        # leaky_relu:
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the block
        :param x: input
        :return: y => output
        """
        y = self.pixNorm(self.lrelu(self.conv_1(x)))
        return y


class PixelwiseNorm(nn.Module):
    def __init__(self):
        super(PixelwiseNorm, self).__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the module
        :param x: input activations volume
        :param alpha: small number for numerical stability
        :return: y => pixel normalized activations
        """
        y = x.pow(2.).mean(dim=1, keepdim=True).add(alpha).sqrt()  # [N1HW]
        y = x / y  # normalize the input x volume
        return y

class _equalized_conv2d(nn.Module):
    """ conv2d with the concept of equalized learning rate
        Args:
            :param c_in: input channels
            :param c_out:  output channels
            :param k_size: kernel size (h, w) should be a tuple or a single integer
            :param stride: stride for conv
            :param pad: padding
            :param bias: whether to use bias or not
    """

    def __init__(self, c_in, c_out, k_size, stride=1, pad=0, bias=True):
        """ constructor for the class """
        from torch.nn.modules.utils import _pair
        from numpy import sqrt, prod

        super().__init__()

        # define the weight and bias if to be used
        self.weight = nn.Parameter(nn.init.normal_(
            torch.empty(c_out, c_in, *_pair(k_size))
        ))

        self.use_bias = bias
        self.stride = stride
        self.pad = pad

        if self.use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out).fill_(0))

        fan_in = prod(_pair(k_size)) * c_in  # value of fan_in
        self.scale = sqrt(2) / sqrt(fan_in)

    def forward(self, x):
        """
        forward pass of the network
        :param x: input
        :return: y => output
        """
        from torch.nn.functional import conv2d

        return conv2d(input=x,
                      weight=self.weight * self.scale,  # scale the weight on runtime
                      bias=self.bias if self.use_bias else None,
                      stride=self.stride,
                      padding=self.pad)

    def extra_repr(self):
        return ", ".join(map(str, self.weight.shape))

class MinibatchStdDev(nn.Module):
    """
    Minibatch standard deviation layer for the discriminator
    """

    def __init__(self):
        """
        derived class constructor
        """
        super().__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the layer
        :param x: input activation volume
        :param alpha: small number for numerical stability
        :return: y => x appended with standard deviation constant map
        """
        batch_size, _, height, width = x.shape

        # [B x C x H x W] Subtract mean over batch.
        y = x - x.mean(dim=0, keepdim=True)

        # [1 x C x H x W]  Calc standard deviation over batch
        y = torch.sqrt(y.pow(2.).mean(dim=0, keepdim=False) + alpha)

        # [1]  Take average over feature_maps and pixels.
        y = y.mean().view(1, 1, 1, 1)

        # [B x 1 x H x W]  Replicate over group and pixels.
        y = y.repeat(batch_size, 1, height, width)

        # [B x C x H x W]  Append as new feature_map.
        y = torch.cat([x, y], 1)

        # return the computed values:
        return y

class DisFinalBlock(nn.Module):
    """ Final block for the Discriminator """

    def __init__(self, in_channels, use_eql=True):
        """
        constructor of the class
        :param in_channels: number of input channels
        :param use_eql: whether to use equalized learning rate
        """
        from torch.nn import LeakyReLU
        from torch.nn import Conv2d

        super().__init__()

        # declare the required modules for forward pass
        self.batch_discriminator = MinibatchStdDev()

        if use_eql:
            self.conv_1 = _equalized_conv2d(in_channels + 1, in_channels, (3, 3),
                                            pad=1, bias=True)
            self.conv_2 = _equalized_conv2d(in_channels, in_channels, (4, 4),stride=2,pad=1,
                                            bias=True)

            # final layer emulates the fully connected layer
            self.conv_3 = _equalized_conv2d(in_channels, 1, (1, 1), bias=True)

        else:
            # modules required:
            self.conv_1 = Conv2d(in_channels + 1, in_channels, (3, 3), padding=1, bias=True)
            self.conv_2 = Conv2d(in_channels, in_channels, (4, 4), bias=True)

            # final conv layer emulates a fully connected layer
            self.conv_3 = Conv2d(in_channels, 1, (1, 1), bias=True)

        # leaky_relu:
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the FinalBlock
        :param x: input
        :return: y => output
        """
        # minibatch_std_dev layer
        y = self.batch_discriminator(x)

        # define the computations
        y = self.lrelu(self.conv_1(y))
        y = self.lrelu(self.conv_2(y))

        # fully connected layer
        y = self.conv_3(y)  # This layer has linear activation

        # flatten the output raw discriminator scores
        return y

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, use_eql=True):
        super(Discriminator, self).__init__()

        self.use_eql = use_eql
        self.in_channels = in_channels

        # modulelist
        self.rgb_to_feature1 = nn.ModuleList([from_rgb(32), from_rgb(64)])
        self.rgb_to_feature2 = nn.ModuleList([from_rgb(32), from_rgb(64)])

        self.layer = _equalized_conv2d(self.in_channels * 2, 64, (1, 1), bias=True)
        # pixel_wise feature normalizer:
        self.pixNorm = PixelwiseNorm()
        # leaky_relu:
        self.lrelu = nn.LeakyReLU(0.2)

        self.layer0 = DisGeneralConvBlock(64, 64, use_eql=self.use_eql)
        # 128*128*32

        self.layer1 = DisGeneralConvBlock(128, 128, use_eql=self.use_eql)
        # 64*64*64

        self.layer2 = DisGeneralConvBlock(256, 256, use_eql=self.use_eql)
        # 32*32*128

        # self.layer3 = DisGeneralConvBlock(512, 512, use_eql=self.use_eql)
        # 16*16*256

        self.layer4 = DisFinalBlock(256, use_eql=self.use_eql)
        # 8*8*512

    def forward(self, img_A, inputs):
        # inputs图片尺寸从小到大
        # Concatenate image and condition image by channels to produce input
        # img_input = torch.cat((img_A, img_B), 1)
        # img_A_128= F.interpolate(img_A, size=[128, 128])
        # img_A_64= F.interpolate(img_A, size=[64, 64])
        # img_A_32= F.interpolate(img_A, size=[32, 32])

        x = torch.cat((img_A[2], inputs[2]), 1)
        y = self.pixNorm(self.lrelu(self.layer(x)))

        y = self.layer0(y)
        # 128*128*64

        x1 = self.rgb_to_feature1[0](img_A[1])
        x2 = self.rgb_to_feature2[0](inputs[1])
        x = torch.cat((x1, x2), 1)
        y = torch.cat((x, y), 1)
        y = self.layer1(y)
        # 64*64*128

        x1 = self.rgb_to_feature1[1](img_A[0])
        x2 = self.rgb_to_feature2[1](inputs[0])
        x = torch.cat((x1, x2), 1)
        y = torch.cat((x, y), 1)
        y = self.layer2(y)
        # 32*32*256

        # x1 = self.rgb_to_feature1[2](img_A[0])
        # x2 = self.rgb_to_feature2[2](inputs[0])
        # x = torch.cat((x1, x2), 1)
        # y = torch.cat((x, y), 1)
        # y = self.layer3(y)
        # 16*16*512

        y = self.layer4(y)
        # 8*8*512

        return y

if __name__ == '__main__':
    d = Discriminator()
    a = torch.zeros(2, 3, 256, 256)
    a1 = torch.zeros(2, 3, 128, 128)
    a2 = torch.zeros(2, 3, 64, 64)
    a3 = torch.zeros(2, 3, 32, 32)
    fake_B = [a3, a2, a1]
    out = d(fake_B, fake_B)
    print(out.shape)