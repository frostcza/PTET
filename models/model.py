import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mmcv.runner import load_checkpoint
import math
# from .modules import *
import cfg


class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]


class LayerNormParallel(nn.Module):
    def __init__(self, num_features):
        super(LayerNormParallel, self).__init__()
        for i in range(cfg.num_parallel):
            setattr(self, 'lrnorm_' + str(i), nn.LayerNorm(num_features, eps=1e-6))

    def forward(self, x_parallel):
        if len(x_parallel) == 1:
            return [getattr(self, 'lrnorm_' + str(2))(x_parallel[0])]
        else:
            return [getattr(self, 'lrnorm_' + str(i))(x) for i, x in enumerate(x_parallel)]


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ModuleParallel(nn.Linear(in_features, hidden_features))
        self.dwconv = DWConv(hidden_features)
        self.act = ModuleParallel(nn.GELU())
        self.fc2 = ModuleParallel(nn.Linear(hidden_features, out_features))
        self.drop = ModuleParallel(nn.Dropout(drop))

        self.exchange = Exchanger()
    
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, mask):
        # x: [B, N, C], mask: [B, N]
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        
        if(cfg.use_exchange):
            if mask is not None:
                fused = x[2]
                x = [x[0], x[1]]
                x = [x_ * mask_.unsqueeze(2) for (x_, mask_) in zip(x, mask)]
                x.append(fused)
                # print(x)
                x = self.exchange(x, mask, mask_threshold_theta = 0.02, mask_threshold_miu = 0.7)

        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = ModuleParallel(nn.Linear(dim, dim, bias=qkv_bias))
        self.kv = ModuleParallel(nn.Linear(dim, dim * 2, bias=qkv_bias))
        self.attn_drop = ModuleParallel(nn.Dropout(attn_drop))
        self.proj = ModuleParallel(nn.Linear(dim, dim))
        self.proj_drop = ModuleParallel(nn.Dropout(proj_drop))

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = ModuleParallel(nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio))
            self.norm = LayerNormParallel(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x[0].shape
        q = self.q(x)
        q = [q_.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) for q_ in q]

        if self.sr_ratio > 1:
            tmp = [x_.permute(0, 2, 1).reshape(B, C, H, W) for x_ in x]
            tmp = self.sr(tmp)
            tmp = [tmp_.reshape(B, C, -1).permute(0, 2, 1) for tmp_ in tmp]
            kv = self.kv(self.norm(tmp))
        else:
            kv = self.kv(x)
        kv = [kv_.reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) for kv_ in kv]
        k, v = [kv_[0] for kv_ in kv], [kv_[1] for kv_ in kv]

        attn = [(q_ @ k_.transpose(-2, -1)) * self.scale for (q_, k_) in zip(q, k)]
        attn = [attn_.softmax(dim=-1) for attn_ in attn]
        attn = self.attn_drop(attn)

        x = [(attn_ @ v_).transpose(1, 2).reshape(B, N, C) for (attn_, v_) in zip(attn, v)]
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class DWConv(nn.Module):
    def __init__(self, dim):
        super(DWConv, self).__init__()
        self.dwconv = ModuleParallel(nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim))

    def forward(self, x, H, W):
        B, N, C = x[0].shape
        x = [x_.transpose(1, 2).view(B, C, H, W) for x_ in x]
        x = self.dwconv(x)
        x = [x_.flatten(2).transpose(1, 2) for x_ in x]

        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., sr_ratio=1):
        super().__init__()
        self.norm1 = LayerNormParallel(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = ModuleParallel(DropPath(drop_path)) if drop_path > 0. else ModuleParallel(nn.Identity())
        self.norm2 = LayerNormParallel(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, mask=None):
        out = self.drop_path(self.attn(self.norm1(x), H, W))
        x = [x_ + out_ for (x_, out_) in zip(x, out)]
        out = self.drop_path(self.mlp(self.norm2(x), H, W, mask=mask))
        x = [x_ + out_ for (x_, out_) in zip(x, out)]
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding """
    def __init__(self, patch_size=3, stride=2, in_chans=3, embed_dim=64):
        super().__init__()
        
        self.proj = ModuleParallel(nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                                   padding=(patch_size // 2, patch_size // 2)))
        self.norm = LayerNormParallel(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x[0].shape
        x = [x_.flatten(2).transpose(1, 2) for x_ in x]
        x = self.norm(x)

        return x, H, W


class PatchUpsample(nn.Module):
    def __init__(self, in_chans, embed_dim):
        super().__init__()

        self.proj = ModuleParallel(nn.Conv2d(in_chans // 4, embed_dim, kernel_size=1))
        self.norm = LayerNormParallel(embed_dim)
        self.pixelshuffle = ModuleParallel(nn.PixelShuffle(2))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.pixelshuffle(x)
        x = self.proj(x)
        B, C, H, W = x[0].shape
        x = [x_.flatten(2).transpose(1, 2) for x_ in x]

        return x, H, W


class Predictor(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.score_nets = nn.ModuleList([nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1),
            nn.GELU()
        ) for _ in range(cfg.predictor_num_parallel)])

    def forward(self, x):
        x = [self.score_nets[i](x[i]) for i in range(cfg.predictor_num_parallel)]
        return x


class Exchanger(nn.Module):
    def __init__(self):
        super(Exchanger, self).__init__()

    def forward(self, x, mask, mask_threshold_theta, mask_threshold_miu):
        # x: [B, N, C], mask: [B, N]
        x0, x1 = torch.zeros_like(x[0]), torch.zeros_like(x[1])

        # mask = [mask_.repeat(x[0].shape[2], 1).transpose(1,0).unsqueeze(0) for mask_ in mask]
        mask = [mask_.unsqueeze(-1).repeat(1, 1, x[0].shape[2]) for mask_ in mask]
        # print(mask)
        mask0greater = (mask[0] >= mask_threshold_theta)
        x0.masked_scatter_(mask0greater, x[0][mask0greater])
        mask0lesser = (mask[0] < mask_threshold_theta)
        x0.masked_scatter_(mask0lesser, x[1][mask0lesser])
        
        mask1greater = (mask[1] >= mask_threshold_theta)
        x1.masked_scatter_(mask1greater, x[1][mask1greater])
        mask1lesser = (mask[1] < mask_threshold_theta)
        x1.masked_scatter_(mask1lesser, x[0][mask1lesser])
        
        # x0[mask[0] >= mask_threshold] = x[0][mask[0] >= mask_threshold]
        # x0[mask[0] < mask_threshold] = x[1][mask[0] < mask_threshold]
        # x1[mask[1] >= mask_threshold] = x[1][mask[1] >= mask_threshold]
        # x1[mask[1] < mask_threshold] = x[0][mask[1] < mask_threshold]
        
        
        fused = x[2]
        mask_miu0 = (mask[0] >= mask_threshold_miu)
        fused.masked_scatter_(mask_miu0, x[0][mask_miu0])
        mask_miu1 = (mask[1] >= mask_threshold_miu)
        fused.masked_scatter_(mask_miu1, x[1][mask_miu1])
        
        return [x0, x1, fused]

class PTET(nn.Module):
    def __init__(self, in_chans=3, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., depths=[1, 1, 1, 1], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.depths = depths

        # transformer downsampling
        self.patch_embed_enc = nn.ModuleList()
        for i in range(4):
            patch_embed = OverlapPatchEmbed(patch_size=3, stride=2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])
            self.patch_embed_enc.append(patch_embed)

        ln2 = nn.Linear(embed_dims[1], embed_dims[0])
        ln3 = nn.Linear(embed_dims[2], embed_dims[1])
        ln4 = nn.Linear(embed_dims[3], embed_dims[2])
        
        ln_b1 = nn.Sequential(nn.Identity())
        ln_b2 = nn.Sequential(ln2)
        ln_b3 = nn.Sequential(ln3, ln2)
        ln_b4 = nn.Sequential(ln4, ln3, ln2)
        
        ln_list = [ln_b1, ln_b2, ln_b3, ln_b4]
        self.mlp_before_predictor = nn.ModuleList(ln_list)
        self.score_predictor = Predictor(embed_dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur_enc = 0
        self.block_enc, self.norm_enc = nn.ModuleList(), nn.ModuleList()
        for idx in [0, 1, 2, 3]:
            block_enc = nn.ModuleList([Block(
                dim=embed_dims[idx], num_heads=num_heads[idx], mlp_ratio=mlp_ratios[idx], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur_enc + i],
                sr_ratio=sr_ratios[idx])
                for i in range(depths[idx])])
            self.block_enc.append(block_enc)
            self.norm_enc.append(LayerNormParallel(embed_dims[idx]))
            cur_enc += depths[idx]

        # transformer upsampling
        self.patch_embed_dec = nn.ModuleList()
        for i in range(4)[::-1]:
            patch_embed = PatchUpsample(in_chans=embed_dims[i], embed_dim=embed_dims[max(0, i - 1)])
            self.patch_embed_dec.append(patch_embed)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur_dec = 0
        self.block_dec, self.norm_dec = nn.ModuleList(), nn.ModuleList()
        for idx in [2, 1, 0]:
            block_dec = nn.ModuleList([Block(
                dim=embed_dims[idx], num_heads=num_heads[idx], mlp_ratio=mlp_ratios[idx], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur_dec + i],
                sr_ratio=sr_ratios[idx])
                for i in range(depths[idx])])
            self.block_dec.append(block_dec)
            self.norm_dec.append(LayerNormParallel(embed_dims[idx]))
            cur_dec += depths[idx]

        self.project = ModuleParallel(nn.Conv2d(embed_dims[idx], 3, 1, 1, 0))
        self.tanh = ModuleParallel(nn.Tanh())
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    def forward(self, x0, x1):
        x_stack = torch.stack((x0, x1), dim=0)
        fused = torch.mean(x_stack, 0)
        x = [x0, x1, fused]
        
        B = x[0].shape[0]
        outs = []
        # x: torch.Size([1, 3, 480, 512])

        count = 0
        # masks = []
        for i in range(len(self.block_enc)):
            x, H, W = self.patch_embed_enc[i](x)
            # print('embed_enc %d:' % i, x[0].shape)
            for idx, blk in enumerate(self.block_enc[i]):
                mask = None
                if idx == len(self.block_enc[i]) - 1:
                    x_to_pred = [x[0], x[1]]
                    feature_before_pred = [self.mlp_before_predictor[count](x_) for x_ in x_to_pred]
                    score = self.score_predictor(feature_before_pred)

                    mask = [score_.reshape(B, -1, 1)[:, :, 0] for score_ in score]  # mask_: [B, N]
                    mask = torch.stack(mask, dim=-1)
                    mask = F.softmax(mask, dim=2)
                    mask = [mask[:,:,0], mask[:,:,1]]
                    # print(mask)
                    
                    count += 1
                x = blk(x, H, W, mask)
            # print('block_enc %d:' % i, x[0].shape)
            x = self.norm_enc[i](x)
            outs.append(x)
            x = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in x]


        for i in range(len(self.block_dec)):
            x, H, W = self.patch_embed_dec[i](x)
            # print('embed_dec %d:' % i, x[0].shape)
            x = [x_ + outs_ for (x_, outs_) in zip(x, outs[::-1][i + 1])]
            for blk in self.block_dec[i]:
                x = blk(x, H, W)
            # print('block_dec %d:' % i, x[0].shape)
            x = self.norm_dec[i](x)
            x = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in x]


        x, H, W = self.patch_embed_dec[3](x)
        # print('embed_enc 4:', x[0].shape)
        x = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in x]

        # print(x[0].shape)
        x = self.tanh(self.project(x))
        
        return x[0], x[1], x[2]


class PTET_for_test(PTET):
    def forward(self, x0, x1):
        x_stack = torch.stack((x0, x1), dim=0)
        fused = torch.mean(x_stack, 0)
        x = [x0, x1, fused]
        
        B = x[0].shape[0]
        outs = []
        # x: torch.Size([1, 3, 480, 512])

        count = 0
        # masks = []
        for i in range(len(self.block_enc)):
            x, H, W = self.patch_embed_enc[i](x)
            # print('embed_enc %d:' % i, x[0].shape)
            for idx, blk in enumerate(self.block_enc[i]):
                mask = None
                if idx == len(self.block_enc[i]) - 1:
                    x_to_pred = [x[0], x[1]]
                    feature_before_pred = [self.mlp_before_predictor[count](x_) for x_ in x_to_pred]
                    score = self.score_predictor(feature_before_pred)

                    mask = [score_.reshape(B, -1, 1)[:, :, 0] for score_ in score]  # mask_: [B, N]
                    mask = torch.stack(mask, dim=-1)
                    mask = F.softmax(mask, dim=2)
                    mask = [mask[:,:,0], mask[:,:,1]]
                    # print(mask)
                    
                    count += 1
                x = blk(x, H, W, mask)
            # print('block_enc %d:' % i, x[0].shape)
            x = self.norm_enc[i](x)
            outs.append([x[2]])
            x = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in x]


        x = [x[2]]
        for i in range(len(self.block_dec)):
            x, H, W = self.patch_embed_dec[i](x)
            # print('embed_dec %d:' % i, x[0].shape)
            x = [x_ + outs_ for (x_, outs_) in zip(x, outs[::-1][i + 1])]
            for blk in self.block_dec[i]:
                x = blk(x, H, W)
            # print('block_dec %d:' % i, x[0].shape)
            x = self.norm_dec[i](x)
            x = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in x]


        x, H, W = self.patch_embed_dec[3](x)
        # print('embed_enc 4:', x[0].shape)
        x = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in x]

        # print(x[0].shape)
        x = self.tanh(self.project(x))
        
        return x[0]
    