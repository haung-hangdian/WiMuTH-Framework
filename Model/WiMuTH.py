import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import trunc_normal_, DropPath
from operator import itemgetter
# from axial_attention.reversible import ReversibleSequence
from torch.autograd.function import Function
from torch.utils.checkpoint import get_device_states, set_device_states

import itertools
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath as TimmDropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from typing import Tuple
# from Model.Samencoder import *
from Model.segment_anything import *
# from Model.deeplabv3_plus import *
# from Model.init_weights import init_weights

# from aunetr2u import DWT, IWT



class Attention_block_r2u(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block_r2u,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi
    

def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    #print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h
class DWT(nn.Module):
    def __init__(self, channels):
        super(DWT, self).__init__()
        self.requires_grad = False
        self.adjust_channels = nn.Conv2d(channels, channels // 4, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = dwt_init(x)
        x = self.adjust_channels(x)  # Adjust the channels
        return x

class IWT(nn.Module):
    def __init__(self, channels):
        super(IWT, self).__init__()
        self.requires_grad = False
        self.adjust_channels = nn.ConvTranspose2d(channels, channels * 4, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.adjust_channels(x)  # Adjust the channels
        x = iwt_init(x)
        return x

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, 
                      stride=stride, dilation=dilation, groups=dim_in)
        self.norm_layer = nn.GroupNorm(4, dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))


class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class conv_block_r2u(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block_r2u,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x
        
class conv_SCEGE(nn.Module):
    def __init__(self, inplanes, planes, stride, padding, dilation, groups, pooling_r, norm_layer=nn.BatchNorm2d):
        super(conv_SCEGE, self).__init__()
        self.conv = conv_block_r2u(ch_in=inplanes, ch_out=planes)
        
        self.shared_conv = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, 
                                     padding=padding, dilation=dilation,
                                     groups=groups, bias=False)

        self.k2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
            self.shared_conv,
            norm_layer(planes)
        )

        self.k3 = nn.Sequential(
            self.shared_conv,
            norm_layer(planes)
        )

        self.k4 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                      padding=padding, dilation=dilation,
                      groups=groups, bias=False),
            norm_layer(planes)
        )

    def forward(self, x):
        identity = self.conv(x)
        out_k2 = self.k2(x)
        
        # 只在尺寸不同时进行插值
        if out_k2.size()[2:] != identity.size()[2:]:
            out_k2 = F.interpolate(out_k2, identity.size()[2:])
        
        out = torch.sigmoid(identity + out_k2)
        out = self.k3(x) * out
        out = self.k4(out)

        return out
       

class DFCH(nn.Module):
    def __init__(self, dim_xh, dim_xl, in_size, b=1 ,gamma=2,d_list=[1,2,5,7]):
        super().__init__()
        self.gamma = gamma
        self.b = b
        t = int(abs(math.log(in_size, 2) + self.b) / self.gamma)
        k = t if t % 2 else t + 1
        self.pre_project = conv_SCEGE(dim_xh, dim_xl, stride=1, padding=2, dilation=2, groups=1, pooling_r=4, norm_layer=nn.BatchNorm2d)
        group_size = dim_xl // 2
        self.g0 = nn.Sequential(
            LayerNorm(normalized_shape=group_size  +1 , data_format='channels_first'),
            nn.Conv2d(group_size  +1  , group_size  +1  , kernel_size=k,stride=1, 
                      padding=(k+(k-1)*(d_list[0]-1))//2, 
                      dilation=d_list[0], groups=group_size  +1  ),
        )
        self.g1 = nn.Sequential(
            LayerNorm(normalized_shape=group_size  +1  , data_format='channels_first'),
            nn.Conv2d(group_size  +1  , group_size +1  , kernel_size=k, stride=1, 
                      padding=(k+(k-1)*(d_list[1]-1))//2, 
                      dilation=d_list[1], groups=group_size  +1  ),
        )
        self.g2 = nn.Sequential(
            LayerNorm(normalized_shape=group_size  +1  , data_format='channels_first'),
            nn.Conv2d(group_size  +1  , group_size  +1 , kernel_size=k, stride=1, 
                      padding=(k+(k-1)*(d_list[2]-1))//2, 
                      dilation=d_list[2], groups=group_size  +1 ),
        )
        self.g3 = nn.Sequential(
            LayerNorm(normalized_shape=group_size  +1 , data_format='channels_first'),
            nn.Conv2d(group_size  +1 , group_size  +1 , kernel_size=k, stride=1, 
                      padding=(k+(k-1)*(d_list[3]-1))//2, 
                      dilation=d_list[3], groups=group_size +1 ),
        )
        self.tail_conv = nn.Sequential(
            LayerNorm(normalized_shape=dim_xl * 2 + 4 , data_format='channels_first'),
            nn.Conv2d(dim_xl * 2 + 4  , dim_xl, 1)
        )##有加sam要加256，上面的要加64
    def forward(self, xh, xl, mask):
        xh = self.pre_project(xh)
        # sam = F.interpolate(sam, size=[xl.size(2), xl.size(3)], mode ='bilinear', align_corners=True)
        xh = F.interpolate(xh, size=[xl.size(2), xl.size(3)], mode ='bilinear', align_corners=True)
        xh_2 = torch.chunk(xh, 4, dim=1)
        xl_2 = torch.chunk(xl, 4, dim=1)
        # sam_2 = torch.chunk(sam, 4, dim=1)
        x0 = self.g0(torch.cat((xh_2[0], xl_2[0], mask), dim=1))
        x1 = self.g1(torch.cat((xh_2[1], xl_2[1], mask), dim=1))
        x2 = self.g2(torch.cat((xh_2[2], xl_2[2], mask), dim=1))
        x3 = self.g3(torch.cat((xh_2[3], xl_2[3], mask), dim=1))
        x = torch.cat((x0,x1,x2,x3), dim=1)
        x = self.tail_conv(x)
        return x
    
class normal_bridge(nn.Module):
    def __init__(self, dim_xh, dim_xl):
        super().__init__()
        self.pre_project = nn.Conv2d(dim_xh, dim_xl, 1)
        self.tail_conv = nn.Sequential(
            LayerNorm(normalized_shape=dim_xl * 2 + 1, data_format='channels_first'),
            nn.Conv2d(dim_xl * 2 + 1, dim_xl, 1)
        )
    def forward(self, xh, xl, mask):
        xh = self.pre_project(xh)
        xh = xh = F.interpolate(xh, size=[xl.size(2), xl.size(3)], mode ='bilinear', align_corners=True)
        x = torch.cat((xh, xl, mask), dim=1)
        x = self.tail_conv(x)
        return x
        
        

class AttentionS(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        
        

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        
        
        
        # original_size = (h, w)
        # print(x.shape)
        # if w > 64 or h > 64:
        #     # 设置新的宽度和高度
        #     new_w = min(w, 64)
        #     new_h = min(h, 64)
            
        #     # 使用双线性插值进行缩放
        #     x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
        
        # print(111111111)
        x = self.norm(x)
        
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)###############这个进行修改，改成ReLU

        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        
        

        return self.to_out(out)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result

class cbam_bridge(nn.Module):
    def __init__(self, dim_xh, dim_xl):
        super().__init__()
        self.pre_project = nn.Conv2d(dim_xh, dim_xl, 1)
        self.cbam = CBAM(dim_xl * 2 + 1)
        self.tail_conv = nn.Sequential(
            LayerNorm(normalized_shape=dim_xl * 2 + 1, data_format='channels_first'),
            nn.Conv2d(dim_xl * 2 + 1, dim_xl, 1)
        )
    def forward(self, xh, xl, mask):
        xh = self.pre_project(xh)
        xh = xh = F.interpolate(xh, size=[xl.size(2), xl.size(3)], mode ='bilinear', align_corners=True)
        x = self.cbam(torch.cat((xh, xl, mask), dim=1))
        x = self.tail_conv(x)
        return x
    
class SEAttention(nn.Module):

    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class SEAtt_bridge(nn.Module):
    def __init__(self, dim_xh, dim_xl):
        super().__init__()
        self.pre_project = nn.Conv2d(dim_xh, dim_xl, 1)
        self.SE = SEAttention(channel=dim_xl *2 + 1)
        self.tail_conv = nn.Sequential(
            LayerNorm(normalized_shape=dim_xl * 2 + 1, data_format='channels_first'),
            nn.Conv2d(dim_xl * 2 + 1, dim_xl, 1)
        )
    def forward(self, xh, xl, mask):
        xh = self.pre_project(xh)
        xh = xh = F.interpolate(xh, size=[xl.size(2), xl.size(3)], mode ='bilinear', align_corners=True)
        x = self.SE(torch.cat((xh, xl, mask), dim=1))
        x = self.tail_conv(x)
        return x

def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc


def stride(x, stride):
    b, c, h, w = x.shape
    return x[:, :, ::stride, ::stride]


def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)


def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)


class ACmix(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1):
        super(ACmix, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.head = head
        self.kernel_att = kernel_att
        self.kernel_conv = kernel_conv
        self.stride = stride
        self.dilation = dilation
        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        self.head_dim = self.out_planes // self.head

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv_p = nn.Conv2d(2, self.head_dim, kernel_size=1)

        self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2
        self.pad_att = torch.nn.ReflectionPad2d(self.padding_att)
        self.unfold = nn.Unfold(kernel_size=self.kernel_att, padding=0, stride=self.stride)
        self.softmax = torch.nn.Softmax(dim=1)

        self.fc = nn.Conv2d(3 * self.head, self.kernel_conv * self.kernel_conv, kernel_size=1, bias=False)
        self.dep_conv = nn.Conv2d(self.kernel_conv * self.kernel_conv * self.head_dim, out_planes,
                                  kernel_size=self.kernel_conv, bias=True, groups=self.head_dim, padding=1,
                                  stride=stride)

        self.reset_parameters()

    def reset_parameters(self):
        init_rate_half(self.rate1)
        init_rate_half(self.rate2)
        kernel = torch.zeros(self.kernel_conv * self.kernel_conv, self.kernel_conv, self.kernel_conv)
        for i in range(self.kernel_conv * self.kernel_conv):
            kernel[i, i // self.kernel_conv, i % self.kernel_conv] = 1.
        kernel = kernel.squeeze(0).repeat(self.out_planes, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.dep_conv.bias = init_rate_0(self.dep_conv.bias)

    def forward(self, x):
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)
        scaling = float(self.head_dim) ** -0.5
        b, c, h, w = q.shape
        h_out, w_out = h // self.stride, w // self.stride

        pe = self.conv_p(position(h, w, x.is_cuda))

        q_att = q.view(b * self.head, self.head_dim, h, w) * scaling
        k_att = k.view(b * self.head, self.head_dim, h, w)
        v_att = v.view(b * self.head, self.head_dim, h, w)

        if self.stride > 1:
            q_att = stride(q_att, self.stride)
            q_pe = stride(pe, self.stride)
        else:
            q_pe = pe

        unfold_k = self.unfold(self.pad_att(k_att)).view(b * self.head, self.head_dim,
                                                         self.kernel_att * self.kernel_att, h_out,
                                                         w_out)  # b*head, head_dim, k_att^2, h_out, w_out
        unfold_rpe = self.unfold(self.pad_att(pe)).view(1, self.head_dim, self.kernel_att * self.kernel_att, h_out,
                                                        w_out)  # 1, head_dim, k_att^2, h_out, w_out

        att = (q_att.unsqueeze(2) * (unfold_k + q_pe.unsqueeze(2) - unfold_rpe)).sum(
            1)  # (b*head, head_dim, 1, h_out, w_out) * (b*head, head_dim, k_att^2, h_out, w_out) -> (b*head, k_att^2, h_out, w_out)
        att = self.softmax(att)

        out_att = self.unfold(self.pad_att(v_att)).view(b * self.head, self.head_dim, self.kernel_att * self.kernel_att,
                                                        h_out, w_out)
        out_att = (att.unsqueeze(1) * out_att).sum(2).view(b, self.out_planes, h_out, w_out)

        f_all = self.fc(torch.cat(
            [q.view(b, self.head, self.head_dim, h * w), k.view(b, self.head, self.head_dim, h * w),
             v.view(b, self.head, self.head_dim, h * w)], 1))
        f_conv = f_all.permute(0, 2, 1, 3).reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])

        out_conv = self.dep_conv(f_conv)

        return self.rate1 * out_att + self.rate2 * out_conv
    
class ACmix_bridge(nn.Module):
    def __init__(self, dim_xh, dim_xl):
        super().__init__()
        self.pre_project = nn.Conv2d(dim_xh, dim_xl, 1)
        self.ACmix = ACmix(in_planes=dim_xl, out_planes=dim_xl)
        self.tail_conv = nn.Sequential(
            LayerNorm(normalized_shape=dim_xl * 2 + 1, data_format='channels_first'),
            nn.Conv2d(dim_xl * 2 + 1, dim_xl, 1)
        )
    def forward(self, xh, xl, mask):
        xh = self.pre_project(xh)
        xh = xh = F.interpolate(xh, size=[xl.size(2), xl.size(3)], mode ='bilinear', align_corners=True)
        x = torch.cat((xh, xl, mask), dim=1)
        x = self.tail_conv(x)
        x = self.ACmix(x)
        
        return x

class Deterministic(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.cpu_state = None
        self.cuda_in_fwd = None
        self.gpu_devices = None
        self.gpu_states = None

    def record_rng(self, *args):
        self.cpu_state = torch.get_rng_state()
        if torch.cuda._initialized:
            self.cuda_in_fwd = True
            self.gpu_devices, self.gpu_states = get_device_states(*args)

    def forward(self, *args, record_rng=False, set_rng=False, **kwargs):
        if record_rng:
            self.record_rng(*args)

        if not set_rng:
            return self.net(*args, **kwargs)

        rng_devices = []
        if self.cuda_in_fwd:
            rng_devices = self.gpu_devices

        with torch.random.fork_rng(devices=rng_devices, enabled=True):
            torch.set_rng_state(self.cpu_state)
            if self.cuda_in_fwd:
                set_device_states(self.gpu_devices, self.gpu_states)
            return self.net(*args, **kwargs)


# heavily inspired by https://github.com/RobinBruegger/RevTorch/blob/master/revtorch/revtorch.py
# once multi-GPU is confirmed working, refactor and send PR back to source
class ReversibleBlock(nn.Module):
    def __init__(self, f, g):
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)

    def forward(self, x, f_args={}, g_args={}):
        x1, x2 = torch.chunk(x, 2, dim=1)
        y1, y2 = None, None

        with torch.no_grad():
            y1 = x1 + self.f(x2, record_rng=self.training, **f_args)
            y2 = x2 + self.g(y1, record_rng=self.training, **g_args)

        return torch.cat([y1, y2], dim=1)

    def backward_pass(self, y, dy, f_args={}, g_args={}):
        y1, y2 = torch.chunk(y, 2, dim=1)
        del y

        dy1, dy2 = torch.chunk(dy, 2, dim=1)
        del dy

        with torch.enable_grad():
            y1.requires_grad = True
            gy1 = self.g(y1, set_rng=True, **g_args)
            torch.autograd.backward(gy1, dy2)

        with torch.no_grad():
            x2 = y2 - gy1
            del y2, gy1

            dx1 = dy1 + y1.grad
            del dy1
            y1.grad = None

        with torch.enable_grad():
            x2.requires_grad = True
            fx2 = self.f(x2, set_rng=True, **f_args)
            torch.autograd.backward(fx2, dx1, retain_graph=True)

        with torch.no_grad():
            x1 = y1 - fx2
            del y1, fx2

            dx2 = dy2 + x2.grad
            del dy2
            x2.grad = None

            x = torch.cat([x1, x2.detach()], dim=1)
            dx = torch.cat([dx1, dx2], dim=1)

        return x, dx


class IrreversibleBlock(nn.Module):
    def __init__(self, f, g):
        super().__init__()
        self.f = f
        self.g = g

    def forward(self, x, f_args, g_args):
        x1, x2 = torch.chunk(x, 2, dim=1)
        y1 = x1 + self.f(x2, **f_args)
        y2 = x2 + self.g(y1, **g_args)
        return torch.cat([y1, y2], dim=1)


class _ReversibleFunction(Function):
    @staticmethod
    def forward(ctx, x, blocks, kwargs):
        ctx.kwargs = kwargs
        for block in blocks:
            x = block(x, **kwargs)
        ctx.y = x.detach()
        ctx.blocks = blocks
        return x

    @staticmethod
    def backward(ctx, dy):
        y = ctx.y
        kwargs = ctx.kwargs
        for block in ctx.blocks[::-1]:
            y, dy = block.backward_pass(y, dy, **kwargs)
        return dy, None, None


class ReversibleSequence(nn.Module):
    def __init__(self, blocks, ):
        super().__init__()
        self.blocks = nn.ModuleList([ReversibleBlock(f, g) for (f, g) in blocks])

    def forward(self, x, arg_route=(True, True), **kwargs):
        f_args, g_args = map(lambda route: kwargs if route else {}, arg_route)
        block_kwargs = {'f_args': f_args, 'g_args': g_args}
        x = torch.cat((x, x), dim=1)
        x = _ReversibleFunction.apply(x, self.blocks, block_kwargs)
        return torch.stack(x.chunk(2, dim=1)).mean(dim=0)


# helper functions

def exists(val):
    return val is not None


def map_el_ind(arr, ind):
    return list(map(itemgetter(ind), arr))


def sort_and_return_indices(arr):
    indices = [ind for ind in range(len(arr))]
    arr = zip(arr, indices)
    arr = sorted(arr)
    return map_el_ind(arr, 0), map_el_ind(arr, 1)


# calculates the permutation to bring the input tensor to something attend-able
# also calculates the inverse permutation to bring the tensor back to its original shape

def calculate_permutations(num_dimensions, emb_dim):
    total_dimensions = num_dimensions + 2
    emb_dim = emb_dim if emb_dim > 0 else (emb_dim + total_dimensions)
    axial_dims = [ind for ind in range(1, total_dimensions) if ind != emb_dim]

    permutations = []

    for axial_dim in axial_dims:
        last_two_dims = [axial_dim, emb_dim]
        dims_rest = set(range(0, total_dimensions)) - set(last_two_dims)
        permutation = [*dims_rest, *last_two_dims]
        permutations.append(permutation)

    return permutations


# helper classes

class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Sequential(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks

    def forward(self, x):
        for f, g in self.blocks:
            x = x + f(x)
            x = x + g(x)
        return x


class PermuteToFrom(nn.Module):
    def __init__(self, permutation, fn):
        super().__init__()
        self.fn = fn
        _, inv_permutation = sort_and_return_indices(permutation)
        self.permutation = permutation
        self.inv_permutation = inv_permutation

    def forward(self, x, **kwargs):
        axial = x.permute(*self.permutation).contiguous()

        shape = axial.shape
        *_, t, d = shape

        # merge all but axial dimension
        axial = axial.reshape(-1, t, d)

        # attention
        axial = self.fn(axial, **kwargs)

        # restore to original shape and permutation
        axial = axial.reshape(*shape)
        axial = axial.permute(*self.inv_permutation).contiguous()
        return axial


# attention

class SelfAttention(nn.Module):
    def __init__(self, dim, heads, dim_heads=None):
        super().__init__()
        self.dim_heads = (dim // heads) if dim_heads is None else dim_heads
        dim_hidden = self.dim_heads * heads

        self.heads = heads
        self.to_q = nn.Linear(dim, dim_hidden, bias=False)
        self.to_kv = nn.Linear(dim, 2 * dim_hidden, bias=False)
        self.to_out = nn.Linear(dim_hidden, dim)

    def forward(self, x, kv=None):
        kv = x if kv is None else kv
        q, k, v = (self.to_q(x), *self.to_kv(kv).chunk(2, dim=-1))

        b, t, d, h, e = *q.shape, self.heads, self.dim_heads

        merge_heads = lambda x: x.reshape(b, -1, h, e).transpose(1, 2).reshape(b * h, -1, e)
        q, k, v = map(merge_heads, (q, k, v))

        dots = torch.einsum('bie,bje->bij', q, k) * (e ** -0.5)
        dots = dots.softmax(dim=-1)
        out = torch.einsum('bij,bje->bie', dots, v)

        out = out.reshape(b, h, -1, e).transpose(1, 2).reshape(b, -1, d)
        out = self.to_out(out)
        return out

     
class MaHS(nn.Module):
    def __init__(self, dim_in, dim_out, x=8, y=8):
        super().__init__()
        
        c_dim_in = dim_in//4
        k_size=3
        pad=(k_size-1) // 2
        
        self.down = nn.Conv2d(c_dim_in, 1, kernel_size=3,stride = 1, padding=1, groups=1)
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),
        )
        
        self.CBAM = CBAM(c_dim_in)
        self.SE = SEAttention(channel=c_dim_in)
        self.att = AttentionS(dim=c_dim_in)############
        self.params_xy = nn.Parameter(torch.Tensor(1, c_dim_in, x, y), requires_grad=True)
        nn.init.ones_(self.params_xy)
        self.conv_xy = nn.Sequential(nn.Conv2d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in), nn.GELU(), nn.Conv2d(c_dim_in, c_dim_in, 1))

        self.params_zx = nn.Parameter(torch.Tensor(1, 1, c_dim_in, x), requires_grad=True)
        nn.init.ones_(self.params_zx)
        self.conv_zx = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in), nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))

        self.params_zy = nn.Parameter(torch.Tensor(1, 1, c_dim_in, y), requires_grad=True)
        nn.init.ones_(self.params_zy)
        self.conv_zy = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in), nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))

        self.dw = nn.Sequential(
                nn.Conv2d(c_dim_in, c_dim_in, 1),
                nn.GELU(),
                nn.Conv2d(c_dim_in, c_dim_in, kernel_size=3, padding=1, groups=c_dim_in)
        )
        
        self.norm1 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
        self.norm2 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
        
        self.ldw = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1, groups=dim_in),
                nn.GELU(),
                nn.Conv2d(dim_in, dim_out, 1),
        )
        
    def forward(self, x):
        x = self.norm1(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        B, C, H, W = x1.size()
        # print(x1.shape)
        #----------xy----------#
        x1 = self.to_patch_embedding(x1)
        params_xy = self.params_xy
        x1 = self.att(x1)
        x1 = rearrange(x1, 'b (h w) c -> b c h w', h = H, w = W)
        x1 = x1 * self.conv_xy(F.interpolate(params_xy, size=x1.shape[2:4],mode='bilinear', align_corners=True))
        #----------zx----------#
        x2 = self.CBAM(x2)
        x2 = x2.permute(0, 3, 1, 2)
        params_zx = self.params_zx
        x2 = x2 * self.conv_zx(F.interpolate(params_zx, size=x2.shape[2:4],mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x2 = x2.permute(0, 2, 3, 1)
        #----------zy----------#
        x3 = self.SE(x3)
        x3 = x3.permute(0, 2, 1, 3)
        params_zy = self.params_zy
        x3 = x3 * self.conv_zy(F.interpolate(params_zy, size=x3.shape[2:4],mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x3 = x3.permute(0, 2, 1, 3)
        #----------dw----------#
        x4 = self.dw(x4)
        #----------concat----------#
        x = torch.cat([x1,x2,x3,x4],dim=1)
        #----------ldw----------#
        x = self.norm2(x)
        x = self.ldw(x)
        return x



class WiMuTH(nn.Module): 
    def __init__(self, input_channels=3, num_classes=3,  c_list=[32,64,128,256,512,1024], bridge=True, gt_ds=True):
        super().__init__()

        self.bridge = bridge
        self.gt_ds = gt_ds
        self.DWT1 = DWT(input_channels*4)
        self.DWT2 = DWT(c_list[0]*4)
        self.DWT3 = DWT(c_list[1]*4)
        self.DWT4 = DWT(c_list[2]*4)
        self.DWT5 = DWT(c_list[3]*4)
        self.DWT6 = DWT(c_list[4]*4)

        self.IWT1 = IWT(c_list[0])
        self.IWT2 = IWT(c_list[1])
        self.IWT3 = IWT(c_list[2])
        self.IWT4 = IWT(c_list[3])
        self.IWT5 = IWT(c_list[4])
        self.IWT6 = IWT(c_list[5])
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.down = nn.AvgPool2d(kernel_size=2, stride=2)

        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 =nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
        ) 
        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
        )
        self.encoder4_normal = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[3], 3, stride=1, padding=1),
        )
        self.encoder5_normal = nn.Sequential(
            nn.Conv2d(c_list[3], c_list[4], 3, stride=1, padding=1),
        )
        self.encoder6_normal = nn.Sequential(
            nn.Conv2d(c_list[4], c_list[5], 3, stride=1, padding=1),
        )
        self.encoder4 = nn.Sequential(
            MaHS(c_list[2], c_list[3]),

        )
        self.encoder5 = nn.Sequential(
            MaHS(c_list[3], c_list[4]),

        )
        self.encoder6 = nn.Sequential(
            MaHS(c_list[4], c_list[5]),

        )

        if bridge: 
            self.GAB1 = DFCH(c_list[1], c_list[0], in_size=512)
            self.GAB2 = DFCH(c_list[2], c_list[1], in_size=256)
            self.GAB3 = DFCH(c_list[3], c_list[2], in_size=128)
            self.GAB4 = DFCH(c_list[4], c_list[3], in_size=64)
            self.GAB5 = DFCH(c_list[5], c_list[4], in_size=32)
            print('group_aggregation_bridge was used')
        if gt_ds:
            self.gt_conv1 = nn.Sequential(nn.Conv2d(c_list[4], 1, 1))
            self.gt_conv2 = nn.Sequential(nn.Conv2d(c_list[3], 1, 1))
            self.gt_conv3 = nn.Sequential(nn.Conv2d(c_list[2], 1, 1))
            self.gt_conv4 = nn.Sequential(nn.Conv2d(c_list[1], 1, 1))
            self.gt_conv5 = nn.Sequential(nn.Conv2d(c_list[0], 1, 1))
            print('gt deep supervision was used')
        
        self.decoder1 = nn.Sequential(
            MaHS(c_list[5], c_list[4]),

        )
        self.decoder1_normal = nn.Sequential(
            nn.Conv2d(c_list[5], c_list[4], 3, stride=1, padding=1),
        ) 
        self.decoder2 = nn.Sequential(
            MaHS(c_list[4], c_list[3]),

        )
        self.decoder2_normal = nn.Sequential(
            nn.Conv2d(c_list[4], c_list[3], 3, stride=1, padding=1),
        ) 
        self.decoder3 = nn.Sequential(
            MaHS(c_list[3], c_list[2]),

        )
        self.decoder3_normal = nn.Sequential(
            nn.Conv2d(c_list[3], c_list[2], 3, stride=1, padding=1),
        )   
        self.decoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1),
        )  
        self.decoder5 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1),
        )  
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        # out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))##没有加小波变换
        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(self.DWT1(x))),1,1))##加了小波变换
        t1 = out # b, c0, H/2, W/2
        
        # out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(self.DWT2(out))),1,1))
        t2 = out # b, c1, H/4, W/4 

        # out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(self.DWT3(out))),1,1))
        t3 = out # b, c2, H/8, W/8
        

        # out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)),2,2))
        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(self.DWT4(out))),1,1))
        t4 = out # b, c3, H/16, W/16
        
        
        # out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)),2,2))
        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(self.DWT5(out))),1,1))

        t5 = out # b, c4, H/32, W/32
        
        # out = F.gelu(self.encoder6(out)) # b, c5, H/32, W/32
        out = F.gelu(self.encoder6(self.DWT6(out))) # b, c5, H/32, W/32

        t6 = out
        # print(t6.shape)
        
        # out5 = F.gelu(self.dbn1(self.decoder1(out))) # b, c4, H/32, W/32  没加小波变换
        out5 = F.gelu(self.dbn1(self.decoder1(self.IWT6(out)))) # b, c4, H/32, W/32  加了小波变换
        gt_pre5 = self.gt_conv1(out5)
        # print(t6.shape)
        # print(t5.shape)
        # print(s3.shape)
        # t5 = self.GAB5(t6, t5, s3, gt_pre5)
        t5 = self.GAB5(t6, t5, gt_pre5)
        # t5 = self.GAB5(t6, t5)
        gt_pre5 = F.interpolate(gt_pre5, scale_factor=32, mode ='bilinear', align_corners=True)
        out5 = torch.add(out5, t5) # b, c4, H/32, W/32
        # out5 = torch.cat((out5,s5), dim=1)
        # out5 = self.sam_out_5(out5)
        

        # out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c3, H/16, W/16  没加小波变换
        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(self.IWT5(out5))),scale_factor=(1,1),mode ='bilinear',align_corners=True)) # b, c3, H/16, W/16  加了小波变换
        gt_pre4 = self.gt_conv2(out4)
        # t4 = self.GAB4(t5, t4, s3, gt_pre4)
        t4 = self.GAB4(t5, t4, gt_pre4)
        # t4 = self.GAB4(t5, t4)
        gt_pre4 = F.interpolate(gt_pre4, scale_factor=16, mode ='bilinear', align_corners=True)
        out4 = torch.add(out4, t4) # b, c3, H/16, W/16
        # out4 = torch.cat((out4,s4), dim=1)
        # out4 = self.sam_out_4(out4)
        

        # out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c2, H/8, W/8  没加小波变换
        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(self.IWT4(out4))),scale_factor=(1,1),mode ='bilinear',align_corners=True)) # b, c2, H/8, W/8  加了小波变换
        gt_pre3 = self.gt_conv3(out3)
        # t3 = self.GAB3(t4, t3, s3, gt_pre3)
        t3 = self.GAB3(t4, t3, gt_pre3)
        # t3 = self.GAB3(t4, t3)
        gt_pre3 = F.interpolate(gt_pre3, scale_factor=8, mode ='bilinear', align_corners=True)
        out3 = torch.add(out3, t3) # b, c2, H/8, W/8
        # out3 = torch.cat((out3,s3), dim=1)
        # out3 = self.sam_out_3(out3) ###SAM加的地方
        
        
        # out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c1, H/4, W/4  没加小波变换
        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(self.IWT3(out3))),scale_factor=(1,1),mode ='bilinear',align_corners=True)) # b, c1, H/4, W/4  加了小波变换
        gt_pre2 = self.gt_conv4(out2)
        # t2 = self.GAB2(t3, t2, s3, gt_pre2)
        t2 = self.GAB2(t3, t2, gt_pre2)
        # t2 = self.GAB2(t3, t2)
        gt_pre2 = F.interpolate(gt_pre2, scale_factor=4, mode ='bilinear', align_corners=True)
        out2 = torch.add(out2, t2) # b, c1, H/4, W/4 
        # out2 = torch.cat((out2,s2), dim=1)
        # out2 = self.sam_out_2(out2)
        
        # out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c0, H/2, W/2  没加小波变换
        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(self.IWT2(out2))),scale_factor=(1,1),mode ='bilinear',align_corners=True)) # b, c0, H/2, W/2  加了小波变换
        gt_pre1 = self.gt_conv5(out1)
        # t1 = self.GAB1(t2, t1, s3, gt_pre1)
        t1 = self.GAB1(t2, t1, gt_pre1)
        # t1 = self.GAB1(t2, t1)
        gt_pre1 = F.interpolate(gt_pre1, scale_factor=2, mode ='bilinear', align_corners=True)
        out1 = torch.add(out1, t1) # b, c0, H/2, W/2
        # out1 = torch.cat((out1,s1), dim=1)
        # out1 = self.sam_out_1(out1)
        
        
        # out0 = F.interpolate(self.final(out1),scale_factor=(2,2),mode ='bilinear',align_corners=True) # b, num_class, H, W  没加小波变换
        out0 = F.interpolate(self.final(self.IWT1(out1)),scale_factor=(1,1),mode ='bilinear',align_corners=True) # b, num_class, H, W   加了小波变换
        
        return (torch.sigmoid(gt_pre5), torch.sigmoid(gt_pre4), torch.sigmoid(gt_pre3), torch.sigmoid(gt_pre2), torch.sigmoid(gt_pre1)), torch.sigmoid((out0))

