import torch.nn as nn
import torch.nn.functional as F
from underwater_model.op import *
from underwater_model.vit import ViT
from underwater_model.aspp import ASPP
from underwater_model.psp import PSPModule
from underwater_model.cait import CaiT
from underwater_model.cct import CCT


class refine_block(nn.Module):
    def __init__(self, n_channels=128, image_size=64):
        super(refine_block, self).__init__()
        # self.aspp =
        self.rfblock_skipconnect = Identity(False)
        self.rfblock_sge = sge_layer(16)
        self.rfblock_eca = eca_layer(3)
        self.rfblock_aspp = ASPP(n_channels, n_channels, [4, 8, 12])
        self.rfblock_psp = PSPModule(n_channels, n_channels)
        self.rfblock_vit = ViT(image_size=image_size, patch_size=int(image_size/8),
                       dim=n_channels, depth=1, heads=1, mlp_dim=n_channels, channels=n_channels)
        self.rfblock_cait = CaiT(image_size=image_size,patch_size=int(image_size/8), num_classes=1000, dim=128,
                         depth=1, cls_depth=2, heads=1, mlp_dim=n_channels, channels=n_channels,
                         dropout=0.1, emb_dropout=0.1, layer_dropout=0.05)
        self.rfblock_cct = CCT(img_size=image_size, n_input_channels=n_channels, embedding_dim=n_channels, n_conv_layers=1,
                       kernel_size=3, stride=1, padding=1, pooling_kernel_size=1,
                       pooling_stride=1, pooling_padding=0, num_layers=1, num_heads=1,
                       mlp_radio=3., num_classes=1000, positional_embedding='learnable')


    def forward(self, x, select):
        if select == 0:
            out = self.rfblock_skipconnect(x)
        elif select == 1:
            out = self.rfblock_sge(x)
        elif select == 2:
            out = self.rfblock_eca(x)
        elif select == 3:
            out = self.rfblock_aspp(x)
        elif select == 4:
            out = self.rfblock_psp(x)
        elif select == 5:
            out = self.rfblock_vit(x)
        elif select == 6:
            out = self.rfblock_cait(x)
        elif select == 7:
            out = self.rfblock_cct(x)
        # out = self.conv3(out)
        return out