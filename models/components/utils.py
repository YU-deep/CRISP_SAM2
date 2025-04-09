# Copyright (c) OpenMMLab. All rights reserved.
# Modified from https://github.com/open-mmlab/mmdetection/blob/main/projects/EfficientDet/efficientdet/utils.py

import math
from typing import Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn.bricks import Swish, build_norm_layer
from torch.nn.init import _calculate_fan_in_and_fan_out, trunc_normal_
from mmdet.registry import MODELS
from mmdet.utils import OptConfigType
from torch.autograd import Function
import torch.nn.functional as F
from torch.nn.modules.utils import _triple, _pair, _single

import adapool_cuda


def variance_scaling_trunc(tensor, gain=1.):
    fan_in, _ = _calculate_fan_in_and_fan_out(tensor)
    gain /= max(1.0, fan_in)
    std = math.sqrt(gain) / .87962566103423978
    return trunc_normal_(tensor, 0., std)


@MODELS.register_module()
class Conv2dSamePadding(nn.Conv2d):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0,
                         dilation, groups, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img_h, img_w = x.size()[-2:]
        kernel_h, kernel_w = self.weight.size()[-2:]
        extra_w = (math.ceil(img_w / self.stride[1]) -
                   1) * self.stride[1] - img_w + kernel_w
        extra_h = (math.ceil(img_h / self.stride[0]) -
                   1) * self.stride[0] - img_h + kernel_h

        left = extra_w // 2
        right = extra_w - left
        top = extra_h // 2
        bottom = extra_h - top
        x = F.pad(x, [left, right, top, bottom])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class MaxPool2dSamePadding(nn.Module):

    def __init__(self,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 stride: Union[int, Tuple[int, int]] = 2,
                 **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride, **kwargs)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2

    def forward(self, x):
        h, w = x.shape[-2:]

        extra_h = (math.ceil(w / self.stride[1]) -
                   1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) -
                   1) * self.stride[0] - h + self.kernel_size[0]

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])
        x = self.pool(x)

        return x


class DepthWiseConvBlock(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            apply_norm: bool = True,
            conv_bn_act_pattern: bool = False,
            norm_cfg: OptConfigType = dict(type='BN', momentum=1e-2, eps=1e-3)
    ) -> None:
        super(DepthWiseConvBlock, self).__init__()
        self.depthwise_conv = Conv2dSamePadding(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            groups=in_channels,
            bias=False)
        self.pointwise_conv = Conv2dSamePadding(
            in_channels, out_channels, kernel_size=1, stride=1)

        self.apply_norm = apply_norm
        if self.apply_norm:
            self.bn = build_norm_layer(norm_cfg, num_features=out_channels)[1]

        self.apply_activation = conv_bn_act_pattern
        if self.apply_activation:
            self.swish = Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        if self.apply_norm:
            x = self.bn(x)
        if self.apply_activation:
            x = self.swish(x)

        return x


class DownChannelBlock(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            apply_norm: bool = True,
            conv_bn_act_pattern: bool = False,
            norm_cfg: OptConfigType = dict(type='BN', momentum=1e-2, eps=1e-3)
    ) -> None:
        super(DownChannelBlock, self).__init__()
        self.down_conv = Conv2dSamePadding(in_channels, out_channels, 1)
        self.apply_norm = apply_norm
        if self.apply_norm:
            self.bn = build_norm_layer(norm_cfg, num_features=out_channels)[1]
        self.apply_activation = conv_bn_act_pattern
        if self.apply_activation:
            self.swish = Swish()

    def forward(self, x):
        x = self.down_conv(x)
        if self.apply_norm:
            x = self.bn(x)
        if self.apply_activation:
            x = self.swish(x)

        return x


# Copied from https://github.com/wenxi-yue/SurgicalSAM/blob/main/surgicalSAM/model.py
class CUDA_ADAPOOL2d(Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, beta, kernel=2, stride=None, return_mask=False):

        assert input.dtype == beta.dtype, '`input` and `beta` are not of the same dtype.'
        beta = torch.clamp(beta, 0., 1.)
        no_batch = False
        if len(input.size()) == 3:
            no_batch = True
            input.unsqueeze_(0)
        B, C, H, W = input.shape
        kernel = _pair(kernel)
        if stride is None:
            stride = kernel
        else:
            stride = _pair(stride)

        oH = (H - kernel[0]) // stride[0] + 1
        oW = (W - kernel[1]) // stride[1] + 1

        output = input.new_zeros((B, C, oH, oW))
        if return_mask:
            mask = input.new_zeros((B, kernel[0] * oH, kernel[1] * oW))
        else:
            mask = input.new_zeros((1))

        adapool_cuda.forward_2d(input.contiguous(), beta, kernel, stride, output, return_mask, mask)
        ctx.save_for_backward(input, beta)
        ctx.kernel = kernel
        ctx.stride = stride
        if return_mask:
            mask_ = mask.detach().clone()
            mask_.requires_grad = False
            CUDA_ADAPOOL2d.mask = mask_
        output = torch.nan_to_num(output)
        if no_batch:
            return output.squeeze_(0)
        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):

        grad_input = torch.zeros_like(ctx.saved_tensors[0])
        grad_beta = torch.zeros_like(ctx.saved_tensors[1])

        saved = [grad_output] + list(ctx.saved_tensors) + [ctx.kernel, ctx.stride, grad_input, grad_beta]
        adapool_cuda.backward_2d(*saved)

        return torch.nan_to_num(saved[-2]), torch.nan_to_num(saved[-1]), None, None, None


class CUDA_ADAPOOL2d_EDSCW(Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, kernel=2, stride=None, return_mask=False):
        no_batch = False
        if len(input.size()) == 3:
            no_batch = True
            input.unsqueeze_(0)
        B, C, H, W = input.shape
        kernel = _pair(kernel)
        if stride is None:
            stride = kernel
        else:
            stride = _pair(stride)

        oH = (H - kernel[0]) // stride[0] + 1
        oW = (W - kernel[1]) // stride[1] + 1

        output = input.new_zeros((B, C, oH, oW))
        if return_mask:
            mask = input.new_zeros((B, kernel[0] * oH, kernel[1] * oW))
        else:
            mask = input.new_zeros((1))

        adapool_cuda.forward_2d_edscw(input.contiguous(), kernel, stride, output, return_mask, mask)
        ctx.save_for_backward(input)
        ctx.kernel = kernel
        ctx.stride = stride
        if return_mask:
            mask_ = mask.detach().clone()
            mask_.requires_grad = False
            CUDA_ADAPOOL2d_EDSCW.mask = mask_
        output = torch.nan_to_num(output)
        if no_batch:
            return output.squeeze_(0)
        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):

        grad_input = torch.zeros_like(ctx.saved_tensors[0])

        saved = [grad_output] + list(ctx.saved_tensors) + [ctx.kernel, ctx.stride, grad_input]
        adapool_cuda.backward_2d_edscw(*saved)

        return torch.nan_to_num(saved[-1]), None, None, None


class CUDA_IDWPOOL2d(Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, kernel=2, stride=None, return_mask=False):
        no_batch = False
        if len(input.size()) == 3:
            no_batch = True
            input.unsqueeze_(0)
        B, C, H, W = input.shape
        kernel = _pair(kernel)
        if stride is None:
            stride = kernel
        else:
            stride = _pair(stride)

        oH = (H - kernel[0]) // stride[0] + 1
        oW = (W - kernel[1]) // stride[1] + 1

        output = input.new_zeros((B, C, oH, oW))
        if return_mask:
            mask = input.new_zeros((B, kernel[0] * oH, kernel[1] * oW))
        else:
            mask = input.new_zeros((1))

        adapool_cuda.forward_2d_idw(input.contiguous(), kernel, stride, output, return_mask, mask)
        ctx.save_for_backward(input)
        ctx.kernel = kernel
        ctx.stride = stride
        if return_mask:
            mask_ = mask.detach().clone()
            mask_.requires_grad = False
            CUDA_ADAPOOL2d_EDSCW.mask = mask_
        output = torch.nan_to_num(output)
        if no_batch:
            return output.squeeze_(0)
        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):

        grad_input = torch.zeros_like(ctx.saved_tensors[0])

        saved = [grad_output] + list(ctx.saved_tensors) + [ctx.kernel, ctx.stride, grad_input]
        adapool_cuda.backward_2d_idw(*saved)

        return torch.nan_to_num(saved[-1]), None, None, None


class CUDA_ADAPOOL2d_EM(Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, kernel=2, stride=None, return_mask=False):

        no_batch = False
        if len(input.size()) == 3:
            no_batch = True
            input.unsqueeze_(0)
        B, C, H, W = input.shape
        kernel = _pair(kernel)
        if stride is None:
            stride = kernel
        else:
            stride = _pair(stride)

        oH = (H - kernel[0]) // stride[0] + 1
        oW = (W - kernel[1]) // stride[1] + 1

        output = input.new_zeros((B, C, oH, oW))
        if return_mask:
            mask = input.new_zeros((B, kernel[0] * oH, kernel[1] * oW))
        else:
            mask = input.new_zeros((1))

        adapool_cuda.forward_2d_em(input.contiguous(), kernel, stride, output, return_mask, mask)
        ctx.save_for_backward(input)
        ctx.kernel = kernel
        ctx.stride = stride
        if return_mask:
            mask_ = mask.detach().clone()
            mask_.requires_grad = False
            CUDA_ADAPOOL2d_EM.mask = mask_
        output = torch.nan_to_num(output)
        if no_batch:
            return output.squeeze_(0)
        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):

        grad_input = torch.zeros_like(ctx.saved_tensors[0])

        saved = [grad_output] + list(ctx.saved_tensors) + [ctx.kernel, ctx.stride, grad_input]
        adapool_cuda.backward_2d_em(*saved)

        return torch.nan_to_num(saved[-1]), None, None, None
