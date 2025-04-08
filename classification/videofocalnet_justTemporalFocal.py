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


class TemporalFocalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, focal_window=3, focal_level=2, focal_factor=2, bias=True, proj_drop=0., attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor

        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.gate = nn.Linear(dim, focal_level + 1, bias=bias)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, T, N, C = x.shape  # x : (Batch, Time, Patches, Channels)
        x = x.permute(0, 2, 1, 3).contiguous()  # Rearrange to (B, N, T, C)
        x = x.view(B * N, T, C)  # Merge batch and spatial dimensions

        # Compute q, k, v
        qkv = self.qkv(x)  # (B*N, T, 3*C)
        qkv = qkv.reshape(B * N, T, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*N, num_heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each tensor: (B*N, num_heads, T, head_dim)

        # Compute gates for each focal level
        gates = torch.sigmoid(self.gate(x).mean(dim=1))  # (B*N, focal_level + 1)

        attn_output = torch.zeros_like(q)  # (B*N, num_heads, T, head_dim)

        for l in range(self.focal_level):
            window_size = self.focal_factor * l + self.focal_window
            attn_mask = self.create_attention_mask(T, window_size, q.device)  # (1, 1, T, T)
            attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # (B*N, num_heads, T, T)
            attn_scores = attn_scores.masked_fill(~attn_mask, torch.finfo(attn_scores.dtype).min)
            attn_probs = self.softmax(attn_scores)
            attn_probs = self.attn_drop(attn_probs)
            output = attn_probs @ v  # (B*N, num_heads, T, head_dim)
            gate_l = gates[:, l].view(B * N, 1, 1, 1)
            attn_output += gate_l * output

        # Global attention without mask
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # (B*N, num_heads, T, T)
        attn_probs = self.softmax(attn_scores)
        attn_probs = self.attn_drop(attn_probs)
        output = attn_probs @ v
        gate_global = gates[:, -1].view(B * N, 1, 1, 1)
        attn_output += gate_global * output

        # Projection and reshaping
        attn_output = attn_output.transpose(1, 2).reshape(B * N, T, C)
        attn_output = self.proj(attn_output)
        attn_output = self.proj_drop(attn_output)

        # Reshape back to (B, T, N, C)
        attn_output = attn_output.view(B, N, T, C)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()  # (B, T, N, C)

        return attn_output

    def create_attention_mask(self, seq_len, window_size, device):
        idxs = torch.arange(seq_len, device=device)
        mask = (idxs.unsqueeze(0) - idxs.unsqueeze(1)).abs() <= window_size // 2
        mask = mask.unsqueeze(0).unsqueeze(0)  # Shape (1, 1, seq_len, seq_len)
        return mask.bool()  # Return a boolean mask

class SpatioTemporalFocalAttention(nn.Module):
    def __init__(self, dim, focal_window, focal_level, focal_factor=2, bias=True, proj_drop=0.,
                use_postln_in_modulation=False, normalize_modulator=False, num_frames=8, num_heads=8):
        super().__init__()

        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.use_postln_in_modulation = use_postln_in_modulation
        self.normalize_modulator = normalize_modulator
        self.num_frames = num_frames

        self.f = nn.Linear(dim, 2*dim + (self.focal_level+1), bias=bias)
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=bias)
        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.focal_layers = nn.ModuleList()

        self.kernel_sizes = []
        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, groups=dim, padding=kernel_size // 2, bias=False),
                    nn.GELU(),
                )
            )
            self.kernel_sizes.append(kernel_size)

        # Add Temporal Focal Attention 
        self.attention_layer = TemporalFocalAttention(dim, num_heads=num_heads, focal_window=focal_window, focal_level=focal_level, focal_factor=focal_factor)

        if self.use_postln_in_modulation:
            self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, H, W, C)
        """
        B, H, W, C = x.shape
        x = rearrange(x, '(b t) h w c -> b t (h w) c', t=self.num_frames, h=H, w=W)
        
        # Temporal Attention
        x = self.attention_layer(x)
        x = rearrange(x, 'b t (h w) c -> (b t) c h w', h=H, w=W)                
        x = x.permute(0, 2, 3, 1).contiguous()
        
        if self.use_postln_in_modulation:
            x = self.ln(x)
        
        # post linear porjection
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}'


class VideoFocalViTBlock(nn.Module):
    r""" Focal Modulation Network Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        focal_level (int): Number of focal levels. 
        focal_window (int): Focal window size at first focal level
        use_layerscale (bool): Whether use layerscale
        layerscale_value (float): Initial layerscale value
        use_postln (bool): Whether use layernorm after modulation
    """

    def __init__(self, dim, input_resolution, mlp_ratio=4., drop=0., drop_path=0., 
                    act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                    focal_level=1, focal_window=3,
                    use_layerscale=False, layerscale_value=1e-4, 
                    use_postln=False, use_postln_in_modulation=False, 
                    normalize_modulator=False, num_frames=8):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.mlp_ratio = mlp_ratio
        self.num_frames = num_frames

        self.focal_window = focal_window
        self.focal_level = focal_level
        self.use_postln = use_postln

        self.norm1 = norm_layer(dim)
        self.modulation = SpatioTemporalFocalAttention(
            dim, proj_drop=drop, focal_window=focal_window, focal_level=self.focal_level, 
            use_postln_in_modulation=use_postln_in_modulation, normalize_modulator=normalize_modulator,
            num_frames=self.num_frames
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.gamma_1 = 1.0
        self.gamma_2 = 1.0    
        if use_layerscale:
            self.gamma_1 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)

        self.H = None
        self.W = None

    def forward(self, x):
        H, W = self.H, self.W
        B, L, C = x.shape
        shortcut = x

        # Focal Modulation
        x = x if self.use_postln else self.norm1(x)
        x = x.view(B, H, W, C)
        x = self.modulation(x).view(B, H * W, C)
        x = x if not self.use_postln else self.norm1(x)

        # FFN
        x = shortcut + self.drop_path(self.gamma_1 * x)
        x = x + self.drop_path(self.gamma_2 * (self.norm2(self.mlp(x)) if self.use_postln else self.mlp(self.norm2(x))))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, " \
               f"mlp_ratio={self.mlp_ratio}"


class BasicLayer(nn.Module):
    """ A basic Focal Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        focal_level (int): Number of focal levels
        focal_window (int): Focal window size at first focal level
        use_layerscale (bool): Whether use layerscale
        layerscale_value (float): Initial layerscale value
        use_postln (bool): Whether use layernorm after modulation
    """

    def __init__(self, dim, out_dim, input_resolution, depth,
                 mlp_ratio=4., drop=0., drop_path=0., norm_layer=nn.LayerNorm, 
                 downsample=None, use_checkpoint=False, 
                 focal_level=1, focal_window=1, 
                 use_conv_embed=False, 
                 use_layerscale=False, layerscale_value=1e-4, 
                 use_postln=False, 
                 use_postln_in_modulation=False, 
                 normalize_modulator=False,
                 num_frames=8):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.num_frames = num_frames
        
        # build blocks
        self.blocks = nn.ModuleList([
            VideoFocalViTBlock(
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
                normalize_modulator=normalize_modulator,
                num_frames=self.num_frames
            )
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(
                img_size=input_resolution, 
                patch_size=2,
                in_chans=dim,
                embed_dim=out_dim,
                use_conv_embed=use_conv_embed, 
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

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=(224, 224), patch_size=4, in_chans=3, embed_dim=96,
                        use_conv_embed=False, norm_layer=None, is_stem=False, tubelet_size=1):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.tubelet_size = tubelet_size

        if use_conv_embed:
            # if we choose to use conv embedding, then we treat the stem and non-stem differently
            if is_stem:
                kernel_size = 7; padding = 2; stride = 4
            else:
                kernel_size = 3; padding = 1; stride = 2
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            if self.tubelet_size == 1:
                self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
            else:
                self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim,
                                kernel_size=(tubelet_size,patch_size[0],patch_size[1]),
                                stride=(tubelet_size,patch_size[0],patch_size[1]))
        
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        if self.tubelet_size == 1:
            B, C, H, W = x.shape

            x = self.proj(x)        
            H, W = x.shape[2:]
            x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
            if self.norm is not None:
                x = self.norm(x)
            return x, H, W
        else:
            B, T, C, H, W = x.shape
            x = x.permute(0,2,1,3,4)
            x = self.proj(x)

            B, C, T, H, W = x.shape
            x = x.permute(0,2,1,3,4)
            x = x.reshape(B*T, C, H, W)

            H, W = x.shape[2:]
            x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
            if self.norm is not None:
                x = self.norm(x)
            return x, H, W


class VideoFocalNet(nn.Module):
    r"""Spatio Temporal Focal Modulation Networks (Video-FocalNets)

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Focal Transformer layer.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        drop_rate (float): Dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False 
        focal_levels (list): How many focal levels at all stages. Note that this excludes the finest-grain level. Default: [1, 1, 1, 1] 
        focal_windows (list): The focal window size at all stages. Default: [7, 5, 3, 1] 
        use_conv_embed (bool): Whether use convolutional embedding. We noted that using convolutional embedding usually improve the performance, but we do not use it by default. Default: False 
        use_layerscale (bool): Whether use layerscale proposed in CaiT. Default: False 
        layerscale_value (float): Value for layer scale. Default: 1e-4 
        use_postln (bool): Whether use layernorm after modulation (it helps stablize training of large models)
    """
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
                use_checkpoint=False,                 
                focal_levels=[2, 2, 2, 2], 
                focal_windows=[3, 3, 3, 3], 
                use_conv_embed=False, 
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
        self.tubelet_size=tubelet_size
        self.num_frames = num_frames//self.tubelet_size
        
        
        # split image into patches using either non-overlapped embedding or overlapped embedding
        self.patch_embed = PatchEmbed(
            img_size=to_2tuple(img_size), 
            patch_size=patch_size, 
            in_chans=in_chans, 
            embed_dim=embed_dim[0], 
            use_conv_embed=use_conv_embed, 
            norm_layer=norm_layer if self.patch_norm else None, 
            is_stem=True,
            tubelet_size=tubelet_size)

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=embed_dim[i_layer], 
                               out_dim=embed_dim[i_layer+1] if (i_layer < self.num_layers - 1) else None,  
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               drop=drop_rate, 
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer, 
                               downsample=PatchEmbed if (i_layer < self.num_layers - 1) else None,
                               focal_level=focal_levels[i_layer], 
                               focal_window=focal_windows[i_layer], 
                               use_conv_embed=use_conv_embed,
                               use_checkpoint=use_checkpoint, 
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
        
        # Defining lateral layers for channel alignment
        self.fuse_dim = embed_dim[1]  # Fixing fuse_dim
        self.lateral_layer1 = nn.Linear(embed_dim[1], self.fuse_dim)  # For stage 1
        self.lateral_layer2 = nn.Linear(embed_dim[2], self.fuse_dim)  # For stage 2
        self.lateral_layer3 = nn.Linear(embed_dim[3], self.fuse_dim)  # For stage 3
        self.lateral_layer4 = nn.Linear(embed_dim[3], self.fuse_dim)  # For stage 4
        self.fuse_smooth = nn.Conv2d(self.fuse_dim, self.fuse_dim, kernel_size=3, stride=1, padding=1)
        
        self.head = nn.Linear(self.num_features + self.fuse_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {''}
    
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {''}

    def forward_features(self, x):
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)

        output_stage1 = None
        output_stage2 = None
        output_stage3 = None
        output_stage4 = None
        H1 = W1 = H2 = W2 = H3 = W3 = H4 = W4 = None

        for idx, layer in enumerate(self.layers):
            x, H, W = layer(x, H, W)  
            if idx == 0:  # First stage
                output_stage1 = x
                H1, W1 = H, W
            if idx == 1:  # Second stage
                output_stage2 = x
                H2, W2 = H, W
            if idx == 2:  # Third stage
                output_stage3 = x
                H3, W3 = H, W
            if idx == 3:  # Fourth stage (last)
                output_stage4 = x
                H4, W4 = H, W

        x = self.norm(x)  # Final normalization
        x = self.avgpool(x.transpose(1, 2))  # Pooling
        x = torch.flatten(x, 1)
        return x, output_stage1, H1, W1, output_stage2, H2, W2, output_stage3, H3, W3, output_stage4, H4, W4

    def forward(self, x):
        b, t, c, h, w = x.size()
        if self.tubelet_size == 1:
            x = x.reshape(-1, c, h, w)

        x, output_stage1, H1, W1, output_stage2, H2, W2, output_stage3, H3, W3, output_stage4, H4, W4 = self.forward_features(x)
        B = x.shape[0]  # Batch size

        # Check if outputs are captured
        if output_stage1 is not None and output_stage2 is not None and output_stage3 is not None and output_stage4 is not None:
            # Project outputs to the same channel dimension
            output_stage1 = self.lateral_layer1(output_stage1)  # [B, L1, fuse_dim]
            output_stage2 = self.lateral_layer2(output_stage2)  # [B, L2, fuse_dim]
            output_stage3 = self.lateral_layer3(output_stage3)  # [B, L3, fuse_dim]
            output_stage4 = self.lateral_layer4(output_stage4)  # [B, L4, fuse_dim]

            # Reshape to [B, C, H, W]
            output_stage1 = output_stage1.transpose(1, 2).view(B, self.fuse_dim, H1, W1)
            output_stage2 = output_stage2.transpose(1, 2).view(B, self.fuse_dim, H2, W2)
            output_stage3 = output_stage3.transpose(1, 2).view(B, self.fuse_dim, H3, W3)
            output_stage4 = output_stage4.transpose(1, 2).view(B, self.fuse_dim, H4, W4)

            # Upsample lower resolution outputs to match the highest resolution
            output_stage4_upsampled = F.interpolate(output_stage4, size=(H1, W1), mode='bilinear', align_corners=False)
            output_stage3_upsampled = F.interpolate(output_stage3, size=(H1, W1), mode='bilinear', align_corners=False)
            output_stage2_upsampled = F.interpolate(output_stage2, size=(H1, W1), mode='bilinear', align_corners=False)

            # Fuse the outputs (addition)
            fused_output = output_stage1 + output_stage2_upsampled + output_stage3_upsampled + output_stage4_upsampled

            # Apply smoothing
            fused_output = self.fuse_smooth(fused_output)

            # Global average pooling to get feature vector
            fused_output = fused_output.view(B, self.fuse_dim, -1).mean(dim=2)  # [B, fuse_dim]

            # Concatenate with the final feature vector x
            x = torch.cat([x, fused_output], dim=1)  # [B, num_features + fuse_dim]

        # Reshape for temporal processing
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
    model = VideoFocalNet(depths=[2, 2, 6, 2], embed_dim=96, **kwargs)
    if pretrained:
        url = model_urls['videofocalnet_tiny']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def videofocalnet_small(pretrained=False, **kwargs):
    model = VideoFocalNet(depths=[2, 2, 18, 2], embed_dim=96, **kwargs)
    if pretrained:
        url = model_urls['videofocalnet_small']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def videofocalnet_base(pretrained=False, **kwargs):
    model = VideoFocalNet(depths=[2, 2, 18, 2], embed_dim=128, **kwargs)
    if pretrained:
        url = model_urls['videofocalnet_base']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


if __name__ == '__main__':
    print('test')
