# --------------------------------------------------------
# FocalNets -- Focal Modulation Networks (Version Conv3D)
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Jianwei Yang (jianwyan@microsoft.com)
# Adapté pour Conv3D par [Votre Nom]
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from timm.data.transforms import str_to_pil_interp
from einops import rearrange

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SpatioTemporalFocalModulation(nn.Module):
    def __init__(self, dim, focal_window, focal_level, focal_factor=2, bias=True, proj_drop=0., use_postln_in_modulation=False, normalize_modulator=False, num_frames=8):
        super().__init__()

        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.use_postln_in_modulation = use_postln_in_modulation
        self.normalize_modulator = normalize_modulator
        self.num_frames = num_frames

        # Replacing Conv2D and Conv1D with Conv3D
        self.focal_layers_3d = nn.ModuleList()
        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window
            self.focal_layers_3d.append(
                nn.Sequential(
                    nn.Conv3d(dim, dim, kernel_size=(kernel_size, kernel_size, kernel_size), 
                              stride=1, groups=dim, padding=kernel_size // 2, bias=False),
                    nn.GELU(),
                )
            )

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.use_postln_in_modulation:
            self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, T, H, W, C)
        """
        B, T, H, W, C = x.shape

        # Rearrange to (B, C, T, H, W) for Conv3D
        x = rearrange(x, 'b t h w c -> b c t h w')

        # Apply Conv3D focal modulation
        ctx_all = 0
        for l in range(self.focal_level):
            ctx = self.focal_layers_3d[l](x)
            ctx_all += ctx

        x_out = self.act(ctx_all)

        # Rearrange back to (B, T, H, W, C)
        x_out = rearrange(x_out, 'b c t h w -> b t h w c')

        if self.use_postln_in_modulation:
            x_out = self.ln(x_out)

        # Apply projection and dropout
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)

        return x_out


class VideoFocalNetBlock(nn.Module):
    def __init__(self, dim, input_resolution, mlp_ratio=4., drop=0., drop_path=0., norm_layer=nn.LayerNorm, focal_window=3, focal_level=1, use_layerscale=False, layerscale_value=1e-4, use_postln=False, use_postln_in_modulation=False, num_frames=8):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.mlp_ratio = mlp_ratio
        self.num_frames = num_frames

        self.norm1 = norm_layer(dim)
        self.modulation = SpatioTemporalFocalModulation(
            dim, focal_window=focal_window, focal_level=focal_level, proj_drop=drop,
            use_postln_in_modulation=use_postln_in_modulation, num_frames=num_frames
        )

        self.drop_path = nn.Identity() if drop_path == 0. else DropPath(drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        self.gamma_1 = 1.0
        self.gamma_2 = 1.0    
        if use_layerscale:
            self.gamma_1 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        B, L, C = x.shape
        shortcut = x

        # Focal Modulation with Conv3D
        x = self.norm1(x)
        x = x.view(B, self.num_frames, self.input_resolution[0], self.input_resolution[1], C)
        x = self.modulation(x).view(B, L, C)

        x = shortcut + self.drop_path(self.gamma_1 * x)
        x = x + self.drop_path(self.gamma_2 * (self.norm2(self.mlp(x))))

        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, out_dim, input_resolution, depth,
                 mlp_ratio=4., drop=0., drop_path=0., norm_layer=nn.LayerNorm, 
                 downsample=None, use_checkpoint=False, 
                 focal_level=1, focal_window=1, 
                 use_layerscale=False, layerscale_value=1e-4, 
                 use_postln=False, 
                 use_postln_in_modulation=False, 
                 num_frames=8):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.num_frames = num_frames
        self.use_layerscale = use_layerscale
        
        # Build blocks
        self.blocks = nn.ModuleList([
            VideoFocalNetBlock(
                dim=dim, 
                input_resolution=input_resolution,
                mlp_ratio=mlp_ratio, 
                drop=drop, 
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                focal_level=focal_level,
                focal_window=focal_window, 
                use_layerscale=use_layerscale, 
                layerscale_value=layerscale_value,
                use_postln=use_postln, 
                use_postln_in_modulation=use_postln_in_modulation, 
                num_frames=self.num_frames  # Removed normalize_modulator
            )
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(
                img_size=input_resolution, 
                patch_size=2,
                in_chans=dim,
                embed_dim=out_dim,
                norm_layer=norm_layer, 
                is_stem=False
            )
        else:
            self.downsample = None

    def forward(self, x, H, W):
        for blk in self.blocks:
            blk.H, blk.W = H, W

            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            x = x.transpose(1, 2).reshape(x.shape[0], -1, H, W)
            x, Ho, Wo = self.downsample(x)
        else:
            Ho, Wo = H, W
        return x, Ho, Wo



class PatchEmbed(nn.Module):
    r""" Patch Embedding for VideoFocalNet using Conv3D for tubelets

    Args:
        img_size (int): Image size. Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        tubelet_size (int): Number of frames in each tubelet for Conv3D.
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
    """
    def __init__(self, img_size=(224, 224), patch_size=4, in_chans=3, embed_dim=96, tubelet_size=1, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.patches_resolution = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=(tubelet_size, patch_size[0], patch_size[1]),
                              stride=(tubelet_size, patch_size[0], patch_size[1]))

        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, T, C, H, W).
        """
        B, T, C, H, W = x.shape
        x = self.proj(x)  # Apply Conv3D to input
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        if self.norm is not None:
            x = self.norm(x)
        return x, H, W



class VideoFocalNet(nn.Module):
    def __init__(self, 
                 img_size=224, 
                 patch_size=4, 
                 in_chans=3, 
                 num_classes=1000, 
                 embed_dim=96, 
                 depths=[2, 2, 6, 2], 
                 mlp_ratio=4., 
                 drop_rate=0., 
                 drop_path_rate=0.1, 
                 norm_layer=nn.LayerNorm, 
                 patch_norm=True, 
                 focal_levels=[2, 2, 2, 2], 
                 focal_windows=[3, 3, 3, 3], 
                 use_layerscale=False, 
                 layerscale_value=1e-4, 
                 use_postln=False, 
                 use_postln_in_modulation=False, 
                 normalize_modulator=False, 
                 num_frames=8, 
                 tubelet_size=1, 
                 **kwargs):
        super().__init__()

        self.num_layers = len(depths)
        embed_dim = [embed_dim * (2 ** i) for i in range(self.num_layers)]
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim[-1]
        self.mlp_ratio = mlp_ratio
        self.tubelet_size = tubelet_size
        self.num_frames = num_frames // self.tubelet_size

        # Patch embedding avec Conv3D pour les tubelets (sans use_conv_embed)
        self.patch_embed = PatchEmbed(
            img_size=to_2tuple(img_size), 
            patch_size=patch_size, 
            in_chans=in_chans, 
            embed_dim=embed_dim[0], 
            norm_layer=norm_layer if self.patch_norm else None, 
            tubelet_size=tubelet_size
        )

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Construction des couches
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=embed_dim[i_layer], 
                out_dim=embed_dim[i_layer+1] if (i_layer < self.num_layers - 1) else None,  
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                  patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                mlp_ratio=self.mlp_ratio,
                drop=drop_rate, 
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer, 
                focal_level=focal_levels[i_layer], 
                focal_window=focal_windows[i_layer], 
                use_layerscale=use_layerscale, 
                layerscale_value=layerscale_value, 
                use_postln=use_postln,
                use_postln_in_modulation=use_postln_in_modulation, 
                normalize_modulator=normalize_modulator,
                num_frames=self.num_frames
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x, H, W = layer(x, H, W)
        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        b,t,c,h,w = x.size()
        if self.tubelet_size==1:
            x =  x.reshape(-1,c,h,w)
        x = self.forward_features(x)
        # Agrégation des frames de la même vidéo BxT, C
        x = x.view(b, self.num_frames, x.shape[-1])
        x = x.mean(dim=1)
        x = self.head(x)
        return x



def build_transforms(img_size, center_crop=False):
    t = []
    if center_crop:
        size = int((256 / 224) * img_size)
        t.append(
            transforms.Resize(size, interpolation=str_to_pil_interp('bicubic'))
        )
        t.append(
            transforms.CenterCrop(img_size)    
        )
    else:
        t.append(
            transforms.Resize(img_size, interpolation=str_to_pil_interp('bicubic'))
        )        
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def build_transforms4display(img_size, center_crop=False):
    t = []
    if center_crop:
        size = int((256 / 224) * img_size)
        t.append(
            transforms.Resize(size, interpolation=str_to_pil_interp('bicubic'))
        )
        t.append(
            transforms.CenterCrop(img_size)    
        )
    else:
        t.append(
            transforms.Resize(img_size, interpolation=str_to_pil_interp('bicubic'))
        )  
    t.append(transforms.ToTensor())
    return transforms.Compose(t)


model_urls = {
    "videofocalnet_tiny": "",
    "videofocalnet_small": "",
    "videofocalnet_base": "",
}


@register_model
def videofocalnet_tiny(pretrained=False, **kwargs):
    model = VideoFocalNet(depths=[2, 2, 6, 2], embed_dim=96, tubelet_size=2, **kwargs)  # tubelet_size=2 pour Conv3D
    if pretrained:
        url = model_urls['videofocalnet_tiny']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def videofocalnet_small(pretrained=False, **kwargs):
    model = VideoFocalNet(depths=[2, 2, 18, 2], embed_dim=96, tubelet_size=2, **kwargs)  # tubelet_size=2 pour Conv3D
    if pretrained:
        url = model_urls['videofocalnet_small']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def videofocalnet_base(pretrained=False, **kwargs):
    model = VideoFocalNet(depths=[2, 2, 18, 2], embed_dim=128, tubelet_size=2, **kwargs)  # tubelet_size=2 pour Conv3D
    if pretrained:
        url = model_urls['videofocalnet_base']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


if __name__ == '__main__':
    # Example usage
    model = videofocalnet_base()
    x = torch.randn(1, 8, 3, 224, 224)  # (B, T, C, H, W)
    output = model(x)
    print(output.shape)  # Should be (1, num_classes)

