from underwater_model.op import *
from underwater_model.restormer import TransformerBlock
class block_choice(nn.Module):
    def __init__(self, n_channels=128):
        super(block_choice, self).__init__()
        self.resblock_skipconnect = Identity(False)
        # self.resblock_sepconv = SepConv(n_channels, n_channels, 3, 1, 1, affine=True, upsample=False)
        # self.resblock_sepconvdouble = SepConvDouble(n_channels, n_channels, 3, 1, 1, affine=True, upsample=False)
        # self.resblock_dilconv = DilConv(n_channels, n_channels, 3, 1, 2, 2, affine=True, upsample=False)
        self.resblock_sge = sge_layer(16)
        # self.resblock_dilconvdouble = DilConvDouble(n_channels, n_channels, 3, 1, 2, 2, affine=True, upsample=False)
        self.resblock_eca = eca_layer(3)
        self.resblock_dil4Conv = DilConv(n_channels, n_channels, 3, 1, 4, 4, affine=True, upsample=False)
        self.resblock_conv = Conv(n_channels, n_channels, 3, 1, 1, affine=True, upsample=False)
        # self.resblock_vit = ViT(image_size=vit_image_size, patch_size=vit_patch_size, dim=128, depth=1, heads=1, mlp_dim=128, channels=128)
        # self.resblock_convdouble = ConvDouble(n_channels, n_channels, 3, 1, 1, affine=True, upsample=False)
        # self.resblock_da = DoubleAttentionLayer(n_channels, n_channels, n_channels)
        self.resblock_da = nn.Sequential(*[TransformerBlock(dim=int(n_channels), num_heads=1, ffn_expansion_factor=2.66,
                         bias=False, LayerNorm_type='WithBias') for i in range(1)])
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
            out = self.resblock_sge(x)
        elif select == 2:
            out = self.resblock_eca(x)
        elif select == 3:
            out = self.resblock_dil4Conv(x)
        elif select == 4:
            out = self.resblock_conv(x)
        elif select == 5:
            out = self.resblock_da(x)
        elif select == 6:
            out = self.resblock_se(x)
        elif select == 7:
            out = self.resblock_sa(x)
        # out = self.conv3(out)
        return out