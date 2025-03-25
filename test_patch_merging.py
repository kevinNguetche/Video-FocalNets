# --------------------------------------------------------
# FocalNets -- Focal Modulation Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Jianwei Yang (jianwyan@microsoft.com)
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


class Mlp(nn.Module):
    """MLP module used in the network."""
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
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
    """Spatio-Temporal Focal Modulation module using Conv3D."""
    def __init__(self, dim, focal_window, focal_level, focal_factor=2,
                 bias=True, proj_drop=0., use_postln_in_modulation=False,
                 normalize_modulator=False):
        super().__init__()

        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.use_postln_in_modulation = use_postln_in_modulation
        self.normalize_modulator = normalize_modulator

        self.f = nn.Linear(dim, 2 * dim + (self.focal_level + 1), bias=bias)
        self.h = nn.Conv3d(dim, dim, kernel_size=1, stride=1, bias=bias)

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.focal_layers = nn.ModuleList()
        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window
            padding = kernel_size // 2
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv3d(dim, dim, kernel_size=kernel_size, stride=1,
                              groups=dim, padding=padding, bias=False),
                    nn.GELU(),
                )
            )

        if self.use_postln_in_modulation:
            self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, T, H, W)
        """
        B, C, T, H, W = x.shape

        # Linear projection
        x = x.permute(0, 2, 3, 4, 1)  # (B, T, H, W, C)
        x = self.f(x)  # (B, T, H, W, 2C + L+1)
        q, ctx, gates = torch.split(x, [C, C, self.focal_level + 1], dim=-1)
        q = q.permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)
        ctx = ctx.permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)
        gates = gates.permute(0, 4, 1, 2, 3)  # (B, L+1, T, H, W)

        # Context aggregation
        ctx_all = 0
        for l in range(self.focal_level):
            ctx_l = self.focal_layers[l](ctx)
            ctx_all += ctx_l * gates[:, l:l+1]

        ctx_global = self.act(ctx.mean(dim=[2, 3, 4], keepdim=True))
        ctx_all += ctx_global * gates[:, self.focal_level:]

        # Normalize context if necessary
        if self.normalize_modulator:
            ctx_all = ctx_all / (self.focal_level + 1)

        # Modulation
        modulator = self.h(ctx_all)
        x_out = q * modulator

        if self.use_postln_in_modulation:
            x_out = x_out.permute(0, 2, 3, 4, 1)  # (B, T, H, W, C)
            x_out = self.ln(x_out)
            x_out = x_out.permute(0, 4, 1, 2, 3)  # Back to (B, C, T, H, W)

        # Post-modulation linear projection
        x_out = x_out.permute(0, 2, 3, 4, 1).contiguous()
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        x_out = x_out.permute(0, 4, 1, 2, 3)  # Back to (B, C, T, H, W)

        return x_out

class VideoFocalNetBlock(nn.Module):
    """VideoFocalNet block with Conv3D."""
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 focal_level=1, focal_window=3, use_layerscale=False,
                 layerscale_value=1e-4, use_postln=False,
                 use_postln_in_modulation=False, normalize_modulator=False):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.use_postln = use_postln

        self.norm1 = norm_layer(dim)
        self.modulation = SpatioTemporalFocalModulation(
            dim, focal_window, focal_level,
            use_postln_in_modulation=use_postln_in_modulation,
            normalize_modulator=normalize_modulator
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        self.gamma_1 = nn.Parameter(layerscale_value * torch.ones(dim),
                                    requires_grad=True) if use_layerscale else 1.0
        self.gamma_2 = nn.Parameter(layerscale_value * torch.ones(dim),
                                    requires_grad=True) if use_layerscale else 1.0

    def forward(self, x):
        """
        x: (B, C, T, H, W)
        """
        shortcut = x

        # Focal modulation
        x = x.permute(0, 2, 3, 4, 1)  # (B, T, H, W, C)
        x = self.norm1(x) if self.use_postln else x
        x = x.permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)
        x = self.modulation(x)
        x = x.permute(0, 2, 3, 4, 1)  # (B, T, H, W, C)
        x = self.norm1(x) if not self.use_postln else x
        x = x.permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)

        x = shortcut + self.drop_path(self.gamma_1 * x)

        # MLP
        x_reshaped = x.permute(0, 2, 3, 4, 1)  # (B, T, H, W, C)
        x_reshaped = self.norm2(x_reshaped)
        x_reshaped = self.mlp(x_reshaped)
        x_reshaped = x_reshaped.permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)

        x = x + self.drop_path(self.gamma_2 * x_reshaped)
        return x

class PatchEmbed(nn.Module):
    """Patch Embedding with Conv3D."""
    def __init__(self, img_size=(224, 224), patch_size=4, in_chans=3,
                 embed_dim=96, norm_layer=None, tubelet_size=1):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size

        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(tubelet_size, patch_size[0], patch_size[1]),
            stride=(tubelet_size, patch_size[0], patch_size[1])
        )

        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        """
        x: (B, C, T, H, W)
        """
        # Padding if necessary
        _, _, T, H, W = x.shape
        pad_t = (self.proj.kernel_size[0] - T % self.proj.stride[0]) % self.proj.stride[0]
        pad_h = (self.proj.kernel_size[1] - H % self.proj.stride[1]) % self.proj.stride[1]
        pad_w = (self.proj.kernel_size[2] - W % self.proj.stride[2]) % self.proj.stride[2]

        if pad_t > 0 or pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t))

        x = self.proj(x)  # (B, embed_dim, T', H', W')
        if self.norm:
            x = x.permute(0, 2, 3, 4, 1).contiguous()
            x = self.norm(x)
            x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

class PatchMerging(nn.Module):
    """Downsample by merging patches, similar to Swin Transformer."""
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)  # Adjusted from 8*dim to 4*dim
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: (B, C, T, H, W)
        """
        B, C, T, H, W = x.shape

        # Decide whether to downsample temporal dimension
        if T >= 2:
            pad_t = T % 2
            if pad_t:
                x = F.pad(x, (0, 0, 0, 0, 0, 1))
            T_out = T // 2
            t0 = x[:, :, 0::2, :, :]
            t1 = x[:, :, 1::2, :, :]
        else:
            T_out = T
            t0 = x
            t1 = x  # Duplicate to maintain the same shape

        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            t0 = F.pad(t0, (0, pad_w, 0, pad_h))
            t1 = F.pad(t1, (0, pad_w, 0, pad_h))

        # Spatial downsampling
        x0 = t0[:, :, :, 0::2, 0::2]
        x1 = t0[:, :, :, 1::2, 0::2]
        x2 = t0[:, :, :, 0::2, 1::2]
        x3 = t0[:, :, :, 1::2, 1::2]

        x = torch.cat([x0, x1, x2, x3], dim=1)  # B, 4*C, T_out, H/2, W/2

        # Apply normalization and reduction
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # B, T_out, H/2, W/2, 4*C
        x = self.norm(x)
        x = self.reduction(x)
        x = x.view(B, T_out, H // 2, W // 2, -1)
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # B, C', T_out, H/2, W/2

        return x

class BasicLayer(nn.Module):
    """Basic layer of the network with Conv3D."""
    def __init__(self, dim, depth, mlp_ratio=4., drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 focal_level=1, focal_window=1, use_layerscale=False,
                 layerscale_value=1e-4, use_postln=False,
                 use_postln_in_modulation=False, normalize_modulator=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        # Build blocks
        self.blocks = nn.ModuleList([
            VideoFocalNetBlock(
                dim=dim,
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
                normalize_modulator=normalize_modulator
            )
            for i in range(depth)
        ])

        self.downsample = downsample(dim=dim, norm_layer=norm_layer) if downsample else None

    def forward(self, x):
        """
        x: (B, C, T, H, W)
        """
        for blk in self.blocks:
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)

        if self.downsample:
            _, _, T, H, W = x.shape
            if T >= 2:
                x = self.downsample(x)
            else:
                x = self.spatial_downsample_only(x)
        return x

    def spatial_downsample_only(self, x):
        # Spatial downsampling only
        B, C, T, H, W = x.shape
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        x0 = x[:, :, :, 0::2, 0::2]
        x1 = x[:, :, :, 1::2, 0::2]
        x2 = x[:, :, :, 0::2, 1::2]
        x3 = x[:, :, :, 1::2, 1::2]

        x = torch.cat([x0, x1, x2, x3], dim=1)  # B, 4*C, T, H/2, W/2

        # Apply normalization and reduction
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # B, T, H/2, W/2, 4*C
        x = self.downsample.norm(x)
        x = self.downsample.reduction(x)
        x = x.view(B, T, H // 2, W // 2, -1)
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # B, C', T, H/2, W/2

        return x

class VideoFocalNet(nn.Module):
    """VideoFocalNet with Conv3D and adjusted downsampling."""
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], mlp_ratio=4., drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, use_checkpoint=False,
                 focal_levels=[2, 2, 2, 2], focal_windows=[3, 3, 3, 3],
                 use_layerscale=False, layerscale_value=1e-4, use_postln=False,
                 use_postln_in_modulation=False, normalize_modulator=False,
                 tubelet_size=1):
        super().__init__()

        self.num_layers = len(depths)
        embed_dims = [embed_dim * (2 ** i) for i in range(self.num_layers)]
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_features = embed_dims[-1]
        self.mlp_ratio = mlp_ratio
        self.tubelet_size = tubelet_size

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dims[0],
            norm_layer=norm_layer,
            tubelet_size=tubelet_size
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            downsample = PatchMerging if (i_layer < self.num_layers - 1) else None
            layer = BasicLayer(
                dim=embed_dims[i_layer],
                depth=depths[i_layer],
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=downsample,
                use_checkpoint=use_checkpoint,
                focal_level=focal_levels[i_layer],
                focal_window=focal_windows[i_layer],
                use_layerscale=use_layerscale,
                layerscale_value=layerscale_value,
                use_postln=use_postln,
                use_postln_in_modulation=use_postln_in_modulation,
                normalize_modulator=normalize_modulator
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.head = nn.Linear(self.num_features, num_classes) \
            if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Weight initialization."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        """
        x: (B, C, T, H, W)
        """
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = x.mean(dim=[2, 3, 4])  # Average over T, H, W
        x = self.norm(x)
        return x

    def forward(self, x):
        """
        x: (B, T, C, H, W)
        """
        x = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        x = self.forward_features(x)
        x = self.head(x)
        return x

if __name__ == '__main__':
    # Example usage with tubelet_size=2
    model = VideoFocalNet(num_classes=1000, tubelet_size=2)
    x = torch.randn(1, 8, 3, 224, 224)  # (B, T, C, H, W)
    output = model(x)
    print(output.shape)  # Should be (1, 1000)

