import math
import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import distributed as dist
from timm.models.layers import drop_path, to_2tuple, trunc_normal_


def generate_attn_mask(num_batches, num_prompts, num_patches, device):
    mask_1d = torch.zeros(num_batches, num_patches + num_prompts) > 0
    mask_1d[:, -num_prompts:] = True
    mask_2d = mask_1d.unsqueeze(1).expand(-1, num_patches + num_prompts, -1)
    mask_2d = mask_2d.to(device)

    return mask_2d


def _norm(x, norm, fast_weights):
    weight = fast_weights.get('weight', norm.weight)
    bias = fast_weights.get('bias', norm.bias)
    norm_shape = norm.normalized_shape
    x = F.layer_norm(x, norm_shape, weight, bias, eps=1e-6)

    return x


def _layer_scale(x, ls, fast_weights):
    gamma = fast_weights.get('gamma', ls.gamma)

    return x * gamma


class LayerScale(nn.Module):
    def __init__(
        self,
        dim,
        init_values=1e-5,
        inplace=False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, fast_weights=dict()):
        fc1_weight = fast_weights.get('fc1.weight', self.fc1.weight)
        fc1_bias = fast_weights.get('fc1.bias', self.fc1.bias)
        fc2_weight = fast_weights.get('fc2.weight', self.fc2.weight)
        fc2_bias = fast_weights.get('fc2.bias', self.fc2.bias)

        # x = self.fc1(x)
        x = F.linear(x, fc1_weight, fc1_bias)

        x = self.act(x)

        # x = self.fc2(x)
        x = F.linear(x, fc2_weight, fc2_bias)

        x = self.drop(x)

        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None, ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim

        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None, fast_weights=dict()):
        qkv_weight = fast_weights.get('qkv.weight', self.qkv.weight)
        qkv_bias = fast_weights.get('qkv.bias', self.qkv.bias)
        proj_weight = fast_weights.get('proj.weight', self.proj.weight)
        proj_bias = fast_weights.get('proj.bias', self.proj.bias)

        B, N, C = x.shape
        # qkv = self.qkv(x)
        qkv = F.linear(x, qkv_weight, qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # mask attention
        if attn_mask is not None:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask
            attn = attn + attn_mask[:, None, :, :]

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        # x = self.proj(x)
        x = F.linear(x, proj_weight, proj_bias)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, attn_head_dim=None, with_scale=False):
        super().__init__()
        self.ls1 = LayerScale(dim) if with_scale else nn.Identity()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim
        )

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.ls2 = LayerScale(dim) if with_scale else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask=None, fast_weights=dict()):
        # assign fast weights
        ls1_weights = dict()
        ls2_weights = dict()
        norm1_weights = dict()
        norm2_weights = dict()
        attn_weights = dict()
        mlp_weights = dict()
        for item in fast_weights:
            if 'ls1.' in item:
                new_item = item.replace('ls1.', '')
                ls1_weights[new_item] = fast_weights[item]
            elif 'ls2.' in item:
                new_item = item.replace('ls2.', '')
                ls2_weights[new_item] = fast_weights[item]
            elif 'norm1.' in item:
                new_item = item.replace('norm1.', '')
                norm1_weights[new_item] = fast_weights[item]
            elif 'norm2.' in item:
                new_item = item.replace('norm2.', '')
                norm2_weights[new_item] = fast_weights[item]
            elif 'attn.' in item:
                new_item = item.replace('attn.', '')
                attn_weights[new_item] = fast_weights[item]
            elif 'mlp.' in item:
                new_item = item.replace('mlp.', '')
                mlp_weights[new_item] = fast_weights[item]

        # forward
        x = x + self.drop_path(
            _layer_scale(
                self.attn(_norm(x, self.norm1, norm1_weights), attn_mask, attn_weights),
                self.ls1,
                ls1_weights
            )
        )
        x = x + self.drop_path(
            _layer_scale(
                self.mlp(_norm(x, self.norm2, norm2_weights), mlp_weights),
                self.ls2,
                ls2_weights)
        )

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, ratio=1):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (ratio ** 2)
        self.patch_shape = (int(img_size[0] // patch_size[0] * ratio), int(img_size[1] // patch_size[1] * ratio))
        self.origin_patch_shape = (int(img_size[0] // patch_size[0]), int(img_size[1] // patch_size[1]))
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=(patch_size[0] // ratio),
                              padding=4 + 2 * (ratio // 2 - 1))

    def forward(self, x, fast_weights=dict()):
        proj_weight = fast_weights.get('proj.weight', self.proj.weight)
        proj_bias = fast_weights.get('proj.bias', self.proj.bias)

        # x = self.proj(x)
        x = F.conv2d(x, proj_weight, proj_bias, stride=self.proj.stride, padding=self.proj.padding)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)

        return x, (Hp, Wp)


class ViT(nn.Module):

    def __init__(self,
                 img_size=224, patch_size=16, in_chans=3, embed_dim=768, embed_ratio=1, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., with_scale=False, norm_layer=None, use_checkpoint=False,
                 frozen_stages=-1, freeze_attn=False, freeze_ffn=False,
                 pre_weights=None):
        # Protect mutable default arguments
        super(ViT, self).__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.patch_embed = PatchEmbed(img_size=img_size,
                                      patch_size=patch_size,
                                      in_chans=in_chans,
                                      embed_dim=embed_dim,
                                      ratio=embed_ratio)
        self.num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, with_scale=with_scale
            )
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)

        if freeze_attn:
            self.freeze_attn()
        if freeze_ffn:
            self.freeze_ffn()
        self.frozen_stages = frozen_stages
        self.freeze_stages(self.frozen_stages)

        self.init_weights(pre_weights)

    def freeze_attn(self):
        for i in range(0, self.depth):
            m = self.blocks[i]
            m.attn.eval()
            m.norm1.eval()
            for param in m.attn.parameters():
                param.requires_grad = False
            for param in m.norm1.parameters():
                param.requires_grad = False

    def freeze_ffn(self):
        # self.pos_embed.requires_grad = False
        # self.patch_embed.eval()
        # for param in self.patch_embed.parameters():
        #     param.requires_grad = False
        for i in range(0, self.depth):
            m = self.blocks[i]
            m.mlp.eval()
            m.norm2.eval()
            for param in m.mlp.parameters():
                param.requires_grad = False
            for param in m.norm2.parameters():
                param.requires_grad = False

    def freeze_stages(self, frozen_stages):
        """Freeze parameters."""
        if frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        for i in range(0, frozen_stages):
            m = self.blocks[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pre_weights=None):
        """Initialize the weights in backbone.
        Args:
            pre_weights (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if pre_weights is None:
            def _init_weights(m):
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)

            self.apply(_init_weights)
        else:
            state_dict = torch.load(pre_weights, map_location='cpu')

            if 'pos_embed' in state_dict:
                pos_embed = state_dict['pos_embed'][:, 1:]
                if pos_embed.shape[1] - 1 != self.patch_embed.num_patches:
                    h = w = int(math.sqrt(pos_embed.shape[1]))
                    re_pos_embed = F.interpolate(pos_embed.transpose(1, 2).reshape(1, -1, h, w),
                                                 size=self.patch_embed.patch_shape,
                                                 mode='bilinear',
                                                 align_corners=False)
                    re_pos_embed = re_pos_embed.reshape(1, -1, self.patch_embed.num_patches).transpose(1, 2)
                    state_dict['pos_embed'] = re_pos_embed

            patch_size = tuple(state_dict['patch_embed.proj.weight'].shape[-2:])
            if patch_size != self.patch_embed.patch_size:
                weight = state_dict['patch_embed.proj.weight']
                re_weight = F.interpolate(weight,
                                          size=self.patch_embed.patch_size,
                                          mode='bilinear',
                                          align_corners=False)
                state_dict['patch_embed.proj.weight'] = re_weight

            u, w = self.load_state_dict(state_dict, strict=False)

            initialized = False
            if dist.is_available():
                initialized = dist.is_initialized()
            if initialized:
                rank = dist.get_rank()
            else:
                rank = 0
            if rank == 0:
                print('ViT: misaligned params during the loading of backbone parameters: {} {}'.format(u, w))

    def get_parameters(self):
        fast_weights = dict()

        patch_embed_params = dict(self.patch_embed.named_parameters())
        for item in patch_embed_params:
            fast_weights['patch_embed.{}'.format(item)] = patch_embed_params[item]

        block_params = dict(self.blocks.named_parameters())
        for item in block_params:
            fast_weights['blocks.{}'.format(item)] = block_params[item]

        norm_params = dict(self.norm.named_parameters())
        for item in norm_params:
            fast_weights['norm.{}'.format(item)] = norm_params[item]

        return fast_weights

    def forward_features(self, x):
        x, (Hp, Wp) = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        B, L, _ = x.shape

        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        x = self.norm(x)
        xp = x.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()

        return xp

    def forward_features_with_embeds(self, x, prompt_embeds, fast_weights):
        # load model parameters
        patch_embed_weights = dict()
        blocks_weights = [dict() for _ in range(len(self.blocks))]
        norm_weights = dict()
        for item in fast_weights:
            if 'patch_embed.' in item:
                new_item = item.replace('patch_embed.', '')
                patch_embed_weights[new_item] = fast_weights[item]

            elif 'blocks.' in item:
                i = int(item.split('.')[1])
                new_item = item.replace('blocks.{}.'.format(i), '')
                blocks_weights[i][new_item] = fast_weights[item]

            elif 'norm.' in item:
                new_item = item.replace('norm.', '')
                norm_weights[new_item] = fast_weights[item]

        # model inference
        x, (Hp, Wp) = self.patch_embed(x, patch_embed_weights)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        B, L, _ = x.shape
        x = torch.cat([x, prompt_embeds], dim=1)

        num_batches, num_prompts = prompt_embeds.shape[:2]
        attn_mask = generate_attn_mask(num_batches, num_prompts, self.num_patches, x.device)

        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, *(x, attn_mask))
            else:
                x = blk(x, attn_mask, blocks_weights[i])

        x = _norm(x, self.norm, norm_weights)

        xp = x[:, 0:L]
        xp = xp.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()
        inc_embeds = x[:, L:]

        return xp, inc_embeds

    def forward(self, x, prompt_embeds=None, fast_weights=None):
        if prompt_embeds is None:
            x = self.forward_features(x)
            return x
        else:
            x, inc_embeds = self.forward_features_with_embeds(x, prompt_embeds, fast_weights)
            return x, inc_embeds

    def train(self, mode=True):
        """Convert the model into training mode."""
        super().train(mode)
        self.freeze_stages(self.frozen_stages)
