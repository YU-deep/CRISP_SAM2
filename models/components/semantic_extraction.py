import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from models.sam2.backbones.image_encoder import ImageEncoder
from models.sam2.sam2_utils import MLP
from transformers import (
    AutoTokenizer,
    CLIPTextModel,
    CLIPTextConfig,
    CLIPVisionConfig,
    CLIPVisionModel,
    AutoFeatureExtractor,
)


class SemanticInteraction(nn.Module):
    def __init__(self, clip_text_ckpt, clip_image_ckp, image_dim, text_dim, num_heads, stem_channel, qkv_bias=False,
                 qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., has_mlp=False):
        super().__init__()
        self.text_encoder = ClipTextEncoder(clip_text_ckpt)
        self.image_encoder = ClipVisionEncoder(clip_image_ckp)
        self.cross_att_one_level_v = CrossAttentionBlock(
            image_dim, num_heads=num_heads, stem_channel=stem_channel,
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path, has_mlp=has_mlp
        )
        self.cross_att_one_level_t = CrossAttentionBlock(
            text_dim, num_heads=num_heads, stem_channel=stem_channel,
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path, has_mlp=has_mlp
        )

        self.in_dim = text_dim + text_dim
        self.co_conv = nn.Conv2d(self.in_dim, self.in_dim // 2, kernel_size=3, stride=1, padding=1, bias=True)

        self.cross_att_two_level_v = CrossAttentionBlock(
            image_dim, num_heads=num_heads, stem_channel=stem_channel,
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path, has_mlp=has_mlp
        )
        self.cross_att_two_level_t = CrossAttentionBlock(
            text_dim, num_heads=num_heads, stem_channel=stem_channel,
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path, has_mlp=has_mlp
        )
        self.linear_1 = nn.Linear(self.in_dim // 2, self.in_dim * 2)
        self.dropout = nn.Dropout(drop)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(self.in_dim * 2, self.linear_1 // 2)

    def forward(self, image, text):
        f_v = self.image_encoder(image)
        f_t = self.text_encoder(text)

        f_vt = self.cross_att_one_level_v(f_v, f_t)
        f_tv = self.cross_att_one_level_t(f_t, f_v)

        f_c = self.co_conv(torch.cat([f_v, f_t], dim=1))

        f_vt_dot = self.cross_att_two_level_v(f_c, f_vt)
        f_tv_dot = self.cross_att_two_level_t(f_c, f_tv)

        raw_out = f_vt_dot + f_tv_dot

        out = self.linear_1(raw_out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.linear_2(out)
        return raw_out, out


class DepthWiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                                   padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, stem_channel=16, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.conv_k = DepthWiseConv(stem_channel, stem_channel, kernel_size=3, padding=1)
        self.conv_v = DepthWiseConv(stem_channel, stem_channel, kernel_size=3, padding=1)

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        B, N, C = x.shape
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1,
                                                                                               3)  # B1C -> B1H(C/H) -> BH1(C/H)

        k = self.conv_k(y)
        k = self.wk(k).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1,
                                                                                  3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.conv_v(y)
        v = self.wv(v).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1,
                                                                                  3)  # BNC -> BNH(C/H) -> BHN(C/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class CrossAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, stem_channel, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., has_mlp=False):
        super().__init__()
        self.norm_x = nn.LayerNorm(dim)
        self.norm_y = nn.LayerNorm(dim)
        self.attn = CrossAttention(
            dim, num_heads=num_heads, stem_channel=stem_channel, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = nn.LayerNorm(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = MLP(dim, mlp_hidden_dim, dim // 4, 3)

    def forward(self, x, y):
        out = self.attn(self.norm_x(x), self.norm_y(y))
        out = self.drop_path(out)
        if self.has_mlp:
            out = self.drop_path(self.mlp(self.norm2(out)))
        return out


class ClipTextEncoder(nn.Module):
    def __init__(self, clip_ckpt):
        """
        :param clip_ckpt: the list of checkpoints of CLIP could be found at: https://huggingface.co/openai
        """
        super().__init__()
        config = CLIPTextConfig()
        self.clip_text_model = CLIPTextModel(config)
        self.tokenizer = AutoTokenizer.from_pretrained(clip_ckpt)
        self.dim_align = nn.Linear(512, 768)
        # freeze text encoder
        for param in self.clip_text_model.parameters():
            param.requires_grad = False

    def organ2tokens(self, organ_names):
        text_list = ['A computerized tomography of a {}.'.format(organ_name) for organ_name in organ_names]
        tokens = self.tokenizer(text_list, padding=True, return_tensors="pt")
        for key in tokens.keys():
            tokens[key] = tokens[key].cuda()
        return tokens

    def forward(self, text):
        if text is None:
            return None
        if type(text) is str:
            text = [text]
        tokens = self.organ2tokens(text)
        clip_outputs = self.clip_text_model(**tokens)
        text_embedding = clip_outputs.pooler_output
        text_embedding = self.dim_align(text_embedding)
        return text_embedding


class ClipVisionEncoder(nn.Module):
    def __init__(self, clip_ckpt):
        """
        :param clip_ckpt: the list of checkpoints of CLIP could be found at: https://huggingface.co/openai
        """
        super().__init__()
        config = CLIPVisionConfig()
        self.clip_vision_model = CLIPVisionModel(config)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(clip_ckpt)
        self.dim_align = nn.Linear(512, 768)
        # freeze text encoder
        for param in self.clip_vision_model.parameters():
            param.requires_grad = False

    def images2features(self, images):
        features = self.feature_extractor(images=images, return_tensors="pt")
        for key in features.keys():
            features[key] = features[key].cuda()
        return features

    def forward(self, images):
        if images is None:
            return None
        if type(images) is not list:
            images = [images]
        features = self.images2features(images)
        clip_outputs = self.clip_vision_model(**features)
        vision_embedding = clip_outputs.pooler_output
        vision_embedding = self.dim_align(vision_embedding)
        return vision_embedding
