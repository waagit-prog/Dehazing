import torch
import torch.nn as nn


from mmcv.runner import BaseModule
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from functools import partial
import warnings



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.norm = nn.BatchNorm2d(in_features)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attn(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.mid_dim = dim//2
        self.dim = dim
        self.act = nn.GELU()
        self.last_fc = nn.Conv2d(self.dim, self.dim, 1)

        self.fc = nn.Conv2d(self.mid_dim, self.mid_dim, 1)
        self.max_pool = nn.MaxPool2d(3, 1, 1)

        self.conv = nn.Conv2d(self.mid_dim, self.mid_dim, 3, 1, 1)

    def forward(self, x):
        self.h, self.w = x.shape[2:]

        lfe = self.act(self.conv(x[:,:self.mid_dim,:,:]))

        hfe = self.act(self.fc(self.max_pool(x[:,self.mid_dim:,:,:])))
        x = torch.cat([lfe, hfe], dim=1)
        x = self.last_fc(x)
        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.norm = nn.BatchNorm2d(d_model)

        # self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.attn = Attn(d_model)
        # self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        # x = self.proj_1(x)
        # x = self.spatial_gating_unit(x)
        x = self.attn(x)
        x = self.norm(x)
        x = self.activation(x)
        # x = self.proj_2(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_cfg=None):
        super().__init__()

        self.attn = Attention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        residual = x.clone()
        x = residual + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(x))
        residual = x.clone()
        x = residual + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, in_channels, out_channels):
        super(OverlapPatchEmbed, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x


class Encoder(BaseModule):
    def __init__(self, in_chans=64, embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[8, 8, 4, 4], drop_rate=0., drop_path_rate=0.5,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[2, 2, 2, 2], num_stages=4,
                 pretrained=None,
                 init_cfg=None,
                 norm_cfg=None):
        super().__init__()

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(in_channels=in_chans if i == 0 else embed_dims[i - 1],
                                            out_channels=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], mlp_ratio=mlp_ratios[i], drop=drop_rate, drop_path=dpr[cur + j], norm_cfg=norm_cfg)
                for j in range(depths[i])])

            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)


    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def forward_features(self, x):
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")

            x = patch_embed(x)
            for blk in block:
                x = blk(x)

            outs.append(x)
        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x[0], x[1], x[2], x[3]


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


if __name__ == '__main__':
    net = Encoder().to('cuda')
    rgb = torch.randn((1, 64, 128, 128)).to('cuda')
    # dep = torch.ones((1, 1, 352, 1216)).to('cuda')
    x1, x2, x3, x4 = net(rgb)
    print(x1.shape)
    print(x2.shape)
    print(x3.shape)
    print(x4.shape)

