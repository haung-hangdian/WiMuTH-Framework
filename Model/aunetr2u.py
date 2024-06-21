import torch
import torch.nn as nn
from torch import cat
from torch.nn import init
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
###############################################################################
# Functions
###############################################################################


def init_weights_r2u(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

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

################小波变换######################
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
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)
    
#####################################
class conv_sparable_block_r2u(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_sparable_block_r2u, self).__init__()
        
        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=ch_in, bias=True)
        
        # Pointwise convolution
        self.pointwise_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=True)
        
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Second depthwise separable convolution
        self.depthwise_conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, groups=ch_out, bias=True)
        self.pointwise_conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=1, stride=1, padding=0, bias=True)
        
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        # Apply first depthwise separable convolution
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        # Apply second depthwise separable convolution
        x = self.depthwise_conv2(x)
        x = self.pointwise_conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        return x



class up_conv_r2u(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv_r2u,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Recurrent_block_r2u(nn.Module):
    def __init__(self,ch_out,t=2):
        super(Recurrent_block_r2u,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1
        
# class RRCNN_block_r2u(nn.Module):
#     def __init__(self,ch_in,ch_out,t=2):
#         super(RRCNN_block_r2u,self).__init__()
#         self.RCNN = nn.Sequential(
#             Recurrent_block_r2u(ch_out,t=t),
#             Recurrent_block_r2u(ch_out,t=t)
#         )
#         self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

#     def forward(self,x):
#         x = self.Conv_1x1(x)
#         x1 = self.RCNN(x)
#         return x+x1

class RRCNN_block_r2u(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block_r2u,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block_r2u(ch_out, t=t),
            Recurrent_block_r2u(ch_out, t=t)
        )
        self.Conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True)
        )
        self.activate = nn.ReLU(inplace=True)
        # self.Conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        x1 = self.Conv(x)
        x2 = self.RCNN(x1)
        out = self.activate(x1 + x2)
        return out



class single_conv_r2u(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(single_conv_r2u,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

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

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, C, H, W,  = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim=3, input_resolution = [512, 512], num_heads = [3, 6, 12, 24], window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B,  C, H, W = x.shape
        L = H * W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.view(B, C, H, W)


        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops
    


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


class U_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block_r2u(ch_in=img_ch,ch_out=32)
        self.Conv2 = conv_block_r2u(ch_in=32,ch_out=64)
        self.Conv3 = conv_block_r2u(ch_in=64,ch_out=128)
        self.Conv4 = conv_block_r2u(ch_in=128,ch_out=256)
        self.Conv5 = conv_block_r2u(ch_in=256,ch_out=512)

        self.Up5 = up_conv_r2u(ch_in=512,ch_out=256)
        self.up_conv_r2u5 = conv_block_r2u(ch_in=512, ch_out=256)

        self.Up4 = up_conv_r2u(ch_in=256,ch_out=128)
        self.up_conv_r2u4 = conv_block_r2u(ch_in=256, ch_out=128)
        
        self.Up3 = up_conv_r2u(ch_in=128,ch_out=64)
        self.up_conv_r2u3 = conv_block_r2u(ch_in=128, ch_out=64)
        
        self.Up2 = up_conv_r2u(ch_in=64,ch_out=32)
        self.up_conv_r2u2 = conv_block_r2u(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)
        
        # 添加softmax层
        self.softmax = nn.Softmax(dim=1)  
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.up_conv_r2u5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.up_conv_r2u4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.up_conv_r2u3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.up_conv_r2u2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = self.sigmoid(d1)
        # d1 = self.softmax(d1)  # 添加softmax操作
        

        return d1
############################WTU_Net################

class WTU_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=3):
        super(WTU_Net,self).__init__()

        self.DWT = DWT()
        self.IWT = IWT()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block_r2u(ch_in=img_ch*4,ch_out=32)
        self.Conv2 = conv_block_r2u(ch_in=32*4,ch_out=64)
        self.Conv3 = conv_block_r2u(ch_in=64*4,ch_out=128)
        self.Conv4 = conv_block_r2u(ch_in=128*4,ch_out=256)
        self.Conv5 = conv_block_r2u(ch_in=256*4,ch_out=512)

        self.Up5 = up_conv_r2u(ch_in=512,ch_out=256*4)
        self.up_conv_r2u5 = conv_block_r2u(ch_in=512, ch_out=256)

        self.Up4 = up_conv_r2u(ch_in=256,ch_out=128*4)
        self.up_conv_r2u4 = conv_block_r2u(ch_in=256, ch_out=128)
        
        self.Up3 = up_conv_r2u(ch_in=128,ch_out=64*4)
        self.up_conv_r2u3 = conv_block_r2u(ch_in=128, ch_out=64)
        
        self.Up2 = up_conv_r2u(ch_in=64,ch_out=32*4)
        self.up_conv_r2u2 = conv_block_r2u(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32,output_ch*4,kernel_size=1,stride=1,padding=0)
        
        # 添加softmax层
        self.softmax = nn.Softmax(dim=1)  
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(self.DWT(x))

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(self.DWT(x2))
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(self.DWT(x3))

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(self.DWT(x4))

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(self.DWT(x5))


        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = self.IWT(d5)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.up_conv_r2u5(d5)
        
        d4 = self.Up4(d5)
        d4 = self.IWT(d4)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.up_conv_r2u4(d4)

        d3 = self.Up3(d4)
        d3 = self.IWT(d3)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.up_conv_r2u3(d3)

        d2 = self.Up2(d3)
        d2 = self.IWT(d2)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.up_conv_r2u2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = self.IWT(d1)
        # d1 = self.sigmoid(d1)
        d1 = self.softmax(d1)  # 添加softmax操作
        

        return d1
###########U_Net_Plus#####################

class U_Net_Plus(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(U_Net_Plus,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block_r2u(ch_in=img_ch,ch_out=32)
        self.Conv2 = conv_block_r2u(ch_in=32,ch_out=64)
        self.Conv3 = conv_block_r2u(ch_in=64,ch_out=128)
        self.Conv4 = conv_block_r2u(ch_in=128,ch_out=256)
        self.Conv5 = conv_block_r2u(ch_in=256,ch_out=512)

        self.Up5 = up_conv_r2u(ch_in=512,ch_out=256)
        self.up_conv_r2u5 = conv_block_r2u(ch_in=512, ch_out=256)

        self.Up4 = up_conv_r2u(ch_in=256,ch_out=128)
        self.up_conv_r2u4 = conv_block_r2u(ch_in=256, ch_out=128)
        
        self.Up3 = up_conv_r2u(ch_in=128,ch_out=64)
        self.up_conv_r2u3 = conv_block_r2u(ch_in=128, ch_out=64)
        
        self.Up2 = up_conv_r2u(ch_in=64,ch_out=32)
        self.up_conv_r2u2 = conv_block_r2u(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)
        
        # 添加softmax层
        self.softmax = nn.Softmax(dim=1)  
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        x6 = self.Up4(x4)
        x6 = torch.cat((x3,x6),dim=1)
        x6 = self.up_conv_r2u4(x6)

        x7 = self.Up3(x3)
        x7 = torch.cat((x2,x7),dim=1)
        x7 = self.up_conv_r2u3(x7)

        x8 = self.Up3(x6)
        x8 = torch.cat((x7,x8),dim=1)
        x8 = self.up_conv_r2u3(x8)

        x9 = self.Up2(x2)
        x9 = torch.cat((x1,x9),dim=1)
        x9 = self.up_conv_r2u2(x9)

        x10 = self.Up2(x7)
        x10 = torch.cat((x9,x10),dim=1)
        x10 = self.up_conv_r2u2(x10)

        x11 = self.Up2(x8)
        x11 = torch.cat((x10,x11),dim=1)
        x11 = self.up_conv_r2u2(x11)


        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.up_conv_r2u5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x6,d4),dim=1)
        d4 = self.up_conv_r2u4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x8,d3),dim=1)
        d3 = self.up_conv_r2u3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x11,d2),dim=1)
        d2 = self.up_conv_r2u2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = self.sigmoid(d1)
        # d1 = self.softmax(d1)  # 添加softmax操作
        

        return d1
    
#############################AttU_Nnet_Plus##################

class AttU_Net_Plus(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(AttU_Net_Plus,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block_r2u(ch_in=img_ch,ch_out=32)
        self.Conv2 = conv_block_r2u(ch_in=32,ch_out=64)
        self.Conv3 = conv_block_r2u(ch_in=64,ch_out=128)
        self.Conv4 = conv_block_r2u(ch_in=128,ch_out=256)
        self.Conv5 = conv_block_r2u(ch_in=256,ch_out=512)

        self.Up5 = up_conv_r2u(ch_in=512,ch_out=256)
        self.Att5 = Attention_block_r2u(F_g=256,F_l=256,F_int=128)
        self.up_conv_r2u5 = conv_block_r2u(ch_in=512, ch_out=256)

        self.Up4 = up_conv_r2u(ch_in=256,ch_out=128)
        self.Att4 = Attention_block_r2u(F_g=128,F_l=128,F_int=64)
        self.up_conv_r2u4 = conv_block_r2u(ch_in=256, ch_out=128)
        
        self.Up3 = up_conv_r2u(ch_in=128,ch_out=64)
        self.Att3 = Attention_block_r2u(F_g=64,F_l=64,F_int=32)
        self.up_conv_r2u3 = conv_block_r2u(ch_in=128, ch_out=64)
        
        self.Up2 = up_conv_r2u(ch_in=64,ch_out=32)
        self.Att2 = Attention_block_r2u(F_g=32,F_l=32,F_int=16)
        self.up_conv_r2u2 = conv_block_r2u(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)
         # 添加softmax层
        self.softmax = nn.Softmax(dim=1)  
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        x6 = self.Up4(x4)
        x3 = self.Att4(g=x6, x=x3)
        x6 = torch.cat((x3,x6),dim=1)
        x6 = self.up_conv_r2u4(x6)

        x7 = self.Up3(x3)
        x2 = self.Att3(g=x7, x=x2)
        x7 = torch.cat((x2,x7),dim=1)
        x7 = self.up_conv_r2u3(x7)

        x8 = self.Up3(x6)
        x7 = self.Att3(g=x8, x=x7)
        x8 = torch.cat((x7,x8),dim=1)
        x8 = self.up_conv_r2u3(x8)

        x9 = self.Up2(x2)
        x1 = self.Att2(g=x9, x=x1)
        x9 = torch.cat((x1,x9),dim=1)
        x9 = self.up_conv_r2u2(x9)

        x10 = self.Up2(x7)
        x9 = self.Att2(g=x10, x=x9)
        x10 = torch.cat((x9,x10),dim=1)
        x10 = self.up_conv_r2u2(x10)

        x11 = self.Up2(x8)
        x10 = self.Att2(g=x11, x=x10)
        x11 = torch.cat((x10,x11),dim=1)
        x11 = self.up_conv_r2u2(x11)


        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.up_conv_r2u5(d5)
        
        d4 = self.Up4(d5)
        x6 = self.Att4(g=d4, x=x6)
        d4 = torch.cat((x6,d4),dim=1)
        d4 = self.up_conv_r2u4(d4)

        d3 = self.Up3(d4)
        x8 = self.Att3(g=d3, x=x8)
        d3 = torch.cat((x8,d3),dim=1)
        d3 = self.up_conv_r2u3(d3)

        d2 = self.Up2(d3)
        x11 = self.Att2(g=d2, x=x11)
        d2 = torch.cat((x11,d2),dim=1)
        d2 = self.up_conv_r2u2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = self.sigmoid(d1)
        # d1 = self.softmax(d1)  # 添加softmax操作
        

        return d1
    
############################R2AttU_Net_Plus#################

class R2AttU_Net_Plus(nn.Module):
    def __init__(self,img_ch=3,output_ch=1, t=2):
        super(R2AttU_Net_Plus,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.RRCNN1 = RRCNN_block_r2u(ch_in=img_ch,ch_out=32,t=t)

        self.RRCNN2 = RRCNN_block_r2u(ch_in=32,ch_out=64,t=t)
        
        self.RRCNN3 = RRCNN_block_r2u(ch_in=64,ch_out=128,t=t)
        
        self.RRCNN4 = RRCNN_block_r2u(ch_in=128,ch_out=256,t=t)
        
        self.RRCNN5 = RRCNN_block_r2u(ch_in=256,ch_out=512,t=t)

        self.Up5 = up_conv_r2u(ch_in=512,ch_out=256)
        self.Att5 = Attention_block_r2u(F_g=256,F_l=256,F_int=128)
        self.Up_RRCNN5 = RRCNN_block_r2u(ch_in=512, ch_out=256,t=t)

        self.Up4 = up_conv_r2u(ch_in=256,ch_out=128)
        self.Att4 = Attention_block_r2u(F_g=128,F_l=128,F_int=64)
        self.Up_RRCNN4 = RRCNN_block_r2u(ch_in=256, ch_out=128,t=t)
        
        self.Up3 = up_conv_r2u(ch_in=128,ch_out=64)
        self.Att3 = Attention_block_r2u(F_g=64,F_l=64,F_int=32)
        self.Up_RRCNN3 = RRCNN_block_r2u(ch_in=128, ch_out=64,t=t)
        
        self.Up2 = up_conv_r2u(ch_in=64,ch_out=32)
        self.Att2 = Attention_block_r2u(F_g=32,F_l=32,F_int=16)
        self.Up_RRCNN2 = RRCNN_block_r2u(ch_in=64, ch_out=32,t=t)

        self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)
         # 添加softmax层
        self.softmax = nn.Softmax(dim=1)  
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        x6 = self.Up4(x4)
        x3 = self.Att4(g=x6, x=x3)
        x6 = torch.cat((x3,x6),dim=1)
        x6 = self.Up_RRCNN4(x6)

        x7 = self.Up3(x3)
        x2 = self.Att3(g=x7, x=x2)
        x7 = torch.cat((x2,x7),dim=1)
        x7 = self.Up_RRCNN3(x7)

        x8 = self.Up3(x6)
        x7 = self.Att3(g=x8, x=x7)
        x8 = torch.cat((x7,x8),dim=1)
        x8 = self.Up_RRCNN3(x8)

        x9 = self.Up2(x2)
        x1 = self.Att2(g=x9, x=x1)
        x9 = torch.cat((x1,x9),dim=1)
        x9 = self.Up_RRCNN2(x9)

        x10 = self.Up2(x7)
        x9 = self.Att2(g=x10, x=x9)
        x10 = torch.cat((x9,x10),dim=1)
        x10 = self.Up_RRCNN2(x10)

        x11 = self.Up2(x8)
        x10 = self.Att2(g=x11, x=x10)
        x11 = torch.cat((x10,x11),dim=1)
        x11 = self.Up_RRCNN2(x11)


        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        x6 = self.Att4(g=d4, x=x6)
        d4 = torch.cat((x6,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x8 = self.Att3(g=d3, x=x8)
        d3 = torch.cat((x8,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x11 = self.Att2(g=d2, x=x11)
        d2 = torch.cat((x11,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = self.sigmoid(d1)
        # d1 = self.softmax(d1)  # 添加softmax操作
        

        return d1

##双重Unet

class W_UNet(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(W_UNet,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1_1 = conv_block_r2u(ch_in=img_ch,ch_out=64)
        self.Conv1_2 = conv_block_r2u(ch_in = output_ch, ch_out=64)
        self.Conv2 = conv_block_r2u(ch_in=64,ch_out=128)
        self.Conv3 = conv_block_r2u(ch_in=128,ch_out=256)
        self.Conv4 = conv_block_r2u(ch_in=256,ch_out=512)
        self.Conv5 = conv_block_r2u(ch_in=512,ch_out=1024)
        

        self.Up5 = up_conv_r2u(ch_in=1024,ch_out=512)
        self.up_conv_r2u5 = conv_block_r2u(ch_in=1024, ch_out=512)

        self.Up4 = up_conv_r2u(ch_in=512,ch_out=256)
        self.up_conv_r2u4 = conv_block_r2u(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv_r2u(ch_in=256,ch_out=128)
        self.up_conv_r2u3 = conv_block_r2u(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv_r2u(ch_in=128,ch_out=64)
        self.up_conv_r2u2 = conv_block_r2u(ch_in=128, ch_out=64)

        self.Conv_1x1_1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)
        self.Conv_1x1_2 = nn.Conv2d(64,3,kernel_size=1,stride=1,padding=0)
        
        # 添加softmax层
        self.softmax = nn.Softmax(dim=1)  

    def forward(self, x):
        # encoding path
        x1 = self.Conv1_1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.up_conv_r2u5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.up_conv_r2u4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.up_conv_r2u3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.up_conv_r2u2(d2)

        d1 = self.Conv_1x1_1(d2)
        # d1 = self.softmax(d1)  # 添加softmax操作

        y1 = self.Conv1_2(d1)

        y2 = self.Conv2(y1)
        y2 = self.Maxpool(y2)

        y3 = self.Conv3(y2)
        y3 = self.Maxpool(y3)

        y4 = self.Conv4(y3)
        y4 = self.Maxpool(y4)

        y5 = self.Conv5(y4)
        y5 = self.Maxpool(y5)

        r5 = self.Up5(y5)
        r5 = torch.cat((y4,r5),dim=1)
        r5 = self.up_conv_r2u5(r5)

        r4 = self.Up4(r5)
        r4 = torch.cat((y3,r4),dim=1)
        r4 = self.up_conv_r2u4(r4)

        r3 = self.Up3(r4)
        r3 = torch.cat((y2,r3),dim=1)
        r3 = self.up_conv_r2u3(r3)

        r2 = self.Up2(r3)
        r2 = torch.cat((y1,r2),dim=1)
        r2 = self.up_conv_r2u2(r2)

        r1 = self.Conv_1x1_2(r2)
        # r1 = self.softmax(r1)  # 添加softmax操作

        return [d1, r1]




class R2U_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1,t=2):
        super(R2U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block_r2u(ch_in=img_ch,ch_out=32,t=t)

        self.RRCNN2 = RRCNN_block_r2u(ch_in=32,ch_out=64,t=t)
        
        self.RRCNN3 = RRCNN_block_r2u(ch_in=64,ch_out=128,t=t)
        
        self.RRCNN4 = RRCNN_block_r2u(ch_in=128,ch_out=256,t=t)
        
        self.RRCNN5 = RRCNN_block_r2u(ch_in=256,ch_out=512,t=t)
        

        self.Up5 = up_conv_r2u(ch_in=512,ch_out=256)
        self.Up_RRCNN5 = RRCNN_block_r2u(ch_in=512, ch_out=256,t=t)
        
        self.Up4 = up_conv_r2u(ch_in=256,ch_out=128)
        self.Up_RRCNN4 = RRCNN_block_r2u(ch_in=256, ch_out=128,t=t)
        
        self.Up3 = up_conv_r2u(ch_in=128,ch_out=64)
        self.Up_RRCNN3 = RRCNN_block_r2u(ch_in=128, ch_out=64,t=t)
        
        self.Up2 = up_conv_r2u(ch_in=64,ch_out=32)
        self.Up_RRCNN2 = RRCNN_block_r2u(ch_in=64, ch_out=32,t=t)

        self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)
         # 添加softmax层
        self.softmax = nn.Softmax(dim=1)  



    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)
        # d1 = self.softmax(d1)   # 添加softmax操作

        return d1



class AttU_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block_r2u(ch_in=img_ch,ch_out=32)
        self.Conv2 = conv_block_r2u(ch_in=32,ch_out=64)
        self.Conv3 = conv_block_r2u(ch_in=64,ch_out=128)
        self.Conv4 = conv_block_r2u(ch_in=128,ch_out=256)
        self.Conv5 = conv_block_r2u(ch_in=256,ch_out=512)

        self.Up5 = up_conv_r2u(ch_in=512,ch_out=256)
        self.Att5 = Attention_block_r2u(F_g=256,F_l=256,F_int=128)
        self.up_conv_r2u5 = conv_block_r2u(ch_in=512, ch_out=256)

        self.Up4 = up_conv_r2u(ch_in=256,ch_out=128)
        self.Att4 = Attention_block_r2u(F_g=128,F_l=128,F_int=64)
        self.up_conv_r2u4 = conv_block_r2u(ch_in=256, ch_out=128)
        
        self.Up3 = up_conv_r2u(ch_in=128,ch_out=64)
        self.Att3 = Attention_block_r2u(F_g=64,F_l=64,F_int=32)
        self.up_conv_r2u3 = conv_block_r2u(ch_in=128, ch_out=64)
        
        self.Up2 = up_conv_r2u(ch_in=64,ch_out=32)
        self.Att2 = Attention_block_r2u(F_g=32,F_l=32,F_int=16)
        self.up_conv_r2u2 = conv_block_r2u(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)
         # 添加softmax层
        self.softmax = nn.Softmax(dim=1) 

    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.up_conv_r2u5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.up_conv_r2u4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.up_conv_r2u3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.up_conv_r2u2(d2)

        d1 = self.Conv_1x1(d2)

        d1 = self.softmax(d1) # 添加softmax操作
        
        return d1


class R2AttU_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1,t=2):
        super(R2AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block_r2u(ch_in=img_ch,ch_out=32,t=t)

        self.RRCNN2 = RRCNN_block_r2u(ch_in=32,ch_out=64,t=t)
        
        self.RRCNN3 = RRCNN_block_r2u(ch_in=64,ch_out=128,t=t)
        
        self.RRCNN4 = RRCNN_block_r2u(ch_in=128,ch_out=256,t=t)
        
        self.RRCNN5 = RRCNN_block_r2u(ch_in=256,ch_out=512,t=t)
        

        self.Up5 = up_conv_r2u(ch_in=512,ch_out=256)
        self.Att5 = Attention_block_r2u(F_g=256,F_l=256,F_int=128)
        self.Up_RRCNN5 = RRCNN_block_r2u(ch_in=512, ch_out=256,t=t)
        
        self.Up4 = up_conv_r2u(ch_in=256,ch_out=128)
        self.Att4 = Attention_block_r2u(F_g=128,F_l=128,F_int=64)
        self.Up_RRCNN4 = RRCNN_block_r2u(ch_in=256, ch_out=128,t=t)
        
        self.Up3 = up_conv_r2u(ch_in=128,ch_out=64)
        self.Att3 = Attention_block_r2u(F_g=64,F_l=64,F_int=32)
        self.Up_RRCNN3 = RRCNN_block_r2u(ch_in=128, ch_out=64,t=t)
        
        self.Up2 = up_conv_r2u(ch_in=64,ch_out=32)
        self.Att2 = Attention_block_r2u(F_g=32,F_l=32,F_int=16)
        self.Up_RRCNN2 = RRCNN_block_r2u(ch_in=64, ch_out=32,t=t)

        self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)
        
        

        # 添加softmax层
        self.softmax = nn.Softmax(dim=1) 


    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)
        
        
   
        d1 = self.softmax(d1)   # 添加softmax操作

        return d1
    
class R2AttU_Net_smx(nn.Module):
    def __init__(self,img_ch=3,output_ch=1,t=2):
        super(R2AttU_Net_smx,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block_r2u(ch_in=img_ch,ch_out=32,t=t)

        self.RRCNN2 = RRCNN_block_r2u(ch_in=32,ch_out=64,t=t)
        
        self.RRCNN3 = RRCNN_block_r2u(ch_in=64,ch_out=128,t=t)
        
        self.RRCNN4 = RRCNN_block_r2u(ch_in=128,ch_out=256,t=t)
        
        self.RRCNN5 = RRCNN_block_r2u(ch_in=256,ch_out=512,t=t)
        

        self.Up5 = up_conv_r2u(ch_in=512,ch_out=256)
        self.Att5 = Attention_block_r2u(F_g=256,F_l=256,F_int=128)
        self.Up_RRCNN5 = RRCNN_block_r2u(ch_in=512, ch_out=256,t=t)
        
        self.Up4 = up_conv_r2u(ch_in=256,ch_out=128)
        self.Att4 = Attention_block_r2u(F_g=128,F_l=128,F_int=64)
        self.Up_RRCNN4 = RRCNN_block_r2u(ch_in=256, ch_out=128,t=t)
        
        self.Up3 = up_conv_r2u(ch_in=128,ch_out=64)
        self.Att3 = Attention_block_r2u(F_g=64,F_l=64,F_int=32)
        self.Up_RRCNN3 = RRCNN_block_r2u(ch_in=128, ch_out=64,t=t)
        
        self.Up2 = up_conv_r2u(ch_in=64,ch_out=32)
        self.Att2 = Attention_block_r2u(F_g=32,F_l=32,F_int=16)
        self.Up_RRCNN2 = RRCNN_block_r2u(ch_in=64, ch_out=32,t=t)

        self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)
        smx = torch.nn.Softmax2d()
        d1 = smx(d1)

        return d1
    
class AttU_Net_small(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(AttU_Net_small,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block_r2u(ch_in=img_ch,ch_out=16)
        self.Conv2 = conv_block_r2u(ch_in=16,ch_out=32)
        self.Conv3 = conv_block_r2u(ch_in=32,ch_out=64)
        self.Conv4 = conv_block_r2u(ch_in=64,ch_out=128)
        self.Conv5 = conv_block_r2u(ch_in=128,ch_out=256)

        self.Up5 = up_conv_r2u(ch_in=256,ch_out=128)
        self.Att5 = Attention_block_r2u(F_g=128,F_l=128,F_int=64)
        self.up_conv_r2u5 = conv_block_r2u(ch_in=256, ch_out=128)

        self.Up4 = up_conv_r2u(ch_in=128,ch_out=64)
        self.Att4 = Attention_block_r2u(F_g=64,F_l=64,F_int=32)
        self.up_conv_r2u4 = conv_block_r2u(ch_in=128, ch_out=64)
        
        self.Up3 = up_conv_r2u(ch_in=64,ch_out=32)
        self.Att3 = Attention_block_r2u(F_g=32,F_l=32,F_int=16)
        self.up_conv_r2u3 = conv_block_r2u(ch_in=64, ch_out=32)
        
        self.Up2 = up_conv_r2u(ch_in=32,ch_out=16)
        self.Att2 = Attention_block_r2u(F_g=16,F_l=16,F_int=8)
        self.up_conv_r2u2 = conv_block_r2u(ch_in=32, ch_out=16)

        self.Conv_1x1 = nn.Conv2d(16,output_ch,kernel_size=1,stride=1,padding=0)
        

    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

#         decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.up_conv_r2u5(d5)
        
        d4 = self.Up4(x4)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.up_conv_r2u4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.up_conv_r2u3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.up_conv_r2u2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
##########################



class DSConv(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, extend_scope, morph,
                 if_offset, device):
        """
        The Dynamic Snake Convolution
        :param in_ch: input channel
        :param out_ch: output channel
        :param kernel_size: the size of kernel
        :param extend_scope: the range to expand (default 1 for this method)
        :param morph: the morphology of the convolution kernel is mainly divided into two types
                        along the x-axis (0) and the y-axis (1) (see the paper for details)
        :param if_offset: whether deformation is required, if it is False, it is the standard convolution kernel
        :param device: set on gpu
        """
        super(DSConv, self).__init__()
        # use the <offset_conv> to learn the deformable offset
        self.offset_conv = nn.Conv2d(in_ch, 2 * kernel_size, 3, padding=1)
        self.bn = nn.BatchNorm2d(2 * kernel_size)
        self.kernel_size = kernel_size

        # two types of the DSConv (along x-axis and y-axis)
        self.dsc_conv_x = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=(kernel_size, 1),  
            stride=(kernel_size, 1),
            padding=0,
        )
        self.dsc_conv_y = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=(1, kernel_size),
            stride=(1, kernel_size),
            padding=0,
        )

        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.extend_scope = extend_scope
        self.morph = morph
        self.if_offset = if_offset
        self.device = device

    def forward(self, f):
        offset = self.offset_conv(f)
        offset = self.bn(offset)
        # We need a range of deformation between -1 and 1 to mimic the snake's swing
        offset = torch.tanh(offset)
        input_shape = f.shape
        dsc = DSC(input_shape, self.kernel_size, self.extend_scope, self.morph,
                  self.device)
        deformed_feature = dsc.deform_conv(f, offset, self.if_offset)
        if self.morph == 0:
            x = self.dsc_conv_x(deformed_feature)
            x = self.gn(x)
            x = self.relu(x)
            return x
        else:
            x = self.dsc_conv_y(deformed_feature)
            x = self.gn(x)
            x = self.relu(x)
            return x


# Core code, for ease of understanding, we mark the dimensions of input and output next to the code
class DSC(object):

    def __init__(self, input_shape, kernel_size, extend_scope, morph, device):
        self.num_points = kernel_size
        self.width = input_shape[2]
        self.height = input_shape[3]
        self.morph = morph
        self.device = device
        self.extend_scope = extend_scope  # offset (-1 ~ 1) * extend_scope

        # define feature map shape
        """
        B: Batch size  C: Channel  W: Width  H: Height
        """
        self.num_batch = input_shape[0]
        self.num_channels = input_shape[1]

    """
    input: offset [B,2*K,W,H]  K: Kernel size (2*K: 2D image, deformation contains <x_offset> and <y_offset>)
    output_x: [B,1,W,K*H]   coordinate map
    output_y: [B,1,K*W,H]   coordinate map
    """

    def _coordinate_map_3D(self, offset, if_offset):
        # offset
        y_offset, x_offset = torch.split(offset, self.num_points, dim=1)

        y_center = torch.arange(0, self.width).repeat([self.height])
        y_center = y_center.reshape(self.height, self.width)
        y_center = y_center.permute(1, 0)
        y_center = y_center.reshape([-1, self.width, self.height])
        y_center = y_center.repeat([self.num_points, 1, 1]).float()
        y_center = y_center.unsqueeze(0)

        x_center = torch.arange(0, self.height).repeat([self.width])
        x_center = x_center.reshape(self.width, self.height)
        x_center = x_center.permute(0, 1)
        x_center = x_center.reshape([-1, self.width, self.height])
        x_center = x_center.repeat([self.num_points, 1, 1]).float()
        x_center = x_center.unsqueeze(0)

        if self.morph == 0:
            """
            Initialize the kernel and flatten the kernel
                y: only need 0
                x: -num_points//2 ~ num_points//2 (Determined by the kernel size)
                !!! The related PPT will be submitted later, and the PPT will contain the whole changes of each step
            """
            y = torch.linspace(0, 0, 1)
            x = torch.linspace(
                -int(self.num_points // 2),
                int(self.num_points // 2),
                int(self.num_points),
            )

            y, x = torch.meshgrid(y, x)
            y_spread = y.reshape(-1, 1)
            x_spread = x.reshape(-1, 1)

            y_grid = y_spread.repeat([1, self.width * self.height])
            y_grid = y_grid.reshape([self.num_points, self.width, self.height])
            y_grid = y_grid.unsqueeze(0)  # [B*K*K, W,H]

            x_grid = x_spread.repeat([1, self.width * self.height])
            x_grid = x_grid.reshape([self.num_points, self.width, self.height])
            x_grid = x_grid.unsqueeze(0)  # [B*K*K, W,H]

            y_new = y_center + y_grid
            x_new = x_center + x_grid

            y_new = y_new.repeat(self.num_batch, 1, 1, 1).to(self.device)
            x_new = x_new.repeat(self.num_batch, 1, 1, 1).to(self.device)

            y_offset_new = y_offset.detach().clone()

            if if_offset:
                y_offset = y_offset.permute(1, 0, 2, 3)
                y_offset_new = y_offset_new.permute(1, 0, 2, 3)
                center = int(self.num_points // 2)

                # The center position remains unchanged and the rest of the positions begin to swing
                # This part is quite simple. The main idea is that "offset is an iterative process"
                y_offset_new[center] = 0
                for index in range(1, center):
                    y_offset_new[center + index] = (y_offset_new[center + index - 1] + y_offset[center + index])
                    y_offset_new[center - index] = (y_offset_new[center - index + 1] + y_offset[center - index])
                y_offset_new = y_offset_new.permute(1, 0, 2, 3).to(self.device)
                y_new = y_new.add(y_offset_new.mul(self.extend_scope))

            y_new = y_new.reshape(
                [self.num_batch, self.num_points, 1, self.width, self.height])
            y_new = y_new.permute(0, 3, 1, 4, 2)
            y_new = y_new.reshape([
                self.num_batch, self.num_points * self.width, 1 * self.height
            ])
            x_new = x_new.reshape(
                [self.num_batch, self.num_points, 1, self.width, self.height])
            x_new = x_new.permute(0, 3, 1, 4, 2)
            x_new = x_new.reshape([
                self.num_batch, self.num_points * self.width, 1 * self.height
            ])
            return y_new, x_new

        else:
            """
            Initialize the kernel and flatten the kernel
                y: -num_points//2 ~ num_points//2 (Determined by the kernel size)
                x: only need 0
            """
            y = torch.linspace(
                -int(self.num_points // 2),
                int(self.num_points // 2),
                int(self.num_points),
            )
            x = torch.linspace(0, 0, 1)

            y, x = torch.meshgrid(y, x)
            y_spread = y.reshape(-1, 1)
            x_spread = x.reshape(-1, 1)

            y_grid = y_spread.repeat([1, self.width * self.height])
            y_grid = y_grid.reshape([self.num_points, self.width, self.height])
            y_grid = y_grid.unsqueeze(0)

            x_grid = x_spread.repeat([1, self.width * self.height])
            x_grid = x_grid.reshape([self.num_points, self.width, self.height])
            x_grid = x_grid.unsqueeze(0)

            y_new = y_center + y_grid
            x_new = x_center + x_grid

            y_new = y_new.repeat(self.num_batch, 1, 1, 1)
            x_new = x_new.repeat(self.num_batch, 1, 1, 1)

            y_new = y_new.to(self.device)
            x_new = x_new.to(self.device)
            x_offset_new = x_offset.detach().clone()

            if if_offset:
                x_offset = x_offset.permute(1, 0, 2, 3)
                x_offset_new = x_offset_new.permute(1, 0, 2, 3)
                center = int(self.num_points // 2)
                x_offset_new[center] = 0
                for index in range(1, center):
                    x_offset_new[center + index] = (x_offset_new[center + index - 1] + x_offset[center + index])
                    x_offset_new[center - index] = (x_offset_new[center - index + 1] + x_offset[center - index])
                x_offset_new = x_offset_new.permute(1, 0, 2, 3).to(self.device)
                x_new = x_new.add(x_offset_new.mul(self.extend_scope))

            y_new = y_new.reshape(
                [self.num_batch, 1, self.num_points, self.width, self.height])
            y_new = y_new.permute(0, 3, 1, 4, 2)
            y_new = y_new.reshape([
                self.num_batch, 1 * self.width, self.num_points * self.height
            ])
            x_new = x_new.reshape(
                [self.num_batch, 1, self.num_points, self.width, self.height])
            x_new = x_new.permute(0, 3, 1, 4, 2)
            x_new = x_new.reshape([
                self.num_batch, 1 * self.width, self.num_points * self.height
            ])
            return y_new, x_new

    """
    input: input feature map [N,C,D,W,H]；coordinate map [N,K*D,K*W,K*H] 
    output: [N,1,K*D,K*W,K*H]  deformed feature map
    """

    def _bilinear_interpolate_3D(self, input_feature, y, x):
        y = y.reshape([-1]).float()
        x = x.reshape([-1]).float()

        zero = torch.zeros([]).int()
        max_y = self.width - 1
        max_x = self.height - 1

        # find 8 grid locations
        y0 = torch.floor(y).int()
        y1 = y0 + 1
        x0 = torch.floor(x).int()
        x1 = x0 + 1

        # clip out coordinates exceeding feature map volume
        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)
        x0 = torch.clamp(x0, zero, max_x)
        x1 = torch.clamp(x1, zero, max_x)

        input_feature_flat = input_feature.flatten()
        input_feature_flat = input_feature_flat.reshape(
            self.num_batch, self.num_channels, self.width, self.height)
        input_feature_flat = input_feature_flat.permute(0, 2, 3, 1)
        input_feature_flat = input_feature_flat.reshape(-1, self.num_channels)
        dimension = self.height * self.width

        base = torch.arange(self.num_batch) * dimension
        base = base.reshape([-1, 1]).float()

        repeat = torch.ones([self.num_points * self.width * self.height
                             ]).unsqueeze(0)
        repeat = repeat.float()

        base = torch.matmul(base, repeat)
        base = base.reshape([-1])

        base = base.to(self.device)

        base_y0 = base + y0 * self.height
        base_y1 = base + y1 * self.height

        # top rectangle of the neighbourhood volume
        index_a0 = base_y0 - base + x0
        index_c0 = base_y0 - base + x1

        # bottom rectangle of the neighbourhood volume
        index_a1 = base_y1 - base + x0
        index_c1 = base_y1 - base + x1

        # get 8 grid values
        value_a0 = input_feature_flat[index_a0.type(torch.int64)].to(self.device)
        value_c0 = input_feature_flat[index_c0.type(torch.int64)].to(self.device)
        value_a1 = input_feature_flat[index_a1.type(torch.int64)].to(self.device)
        value_c1 = input_feature_flat[index_c1.type(torch.int64)].to(self.device)

        # find 8 grid locations
        y0 = torch.floor(y).int()
        y1 = y0 + 1
        x0 = torch.floor(x).int()
        x1 = x0 + 1

        # clip out coordinates exceeding feature map volume
        y0 = torch.clamp(y0, zero, max_y + 1)
        y1 = torch.clamp(y1, zero, max_y + 1)
        x0 = torch.clamp(x0, zero, max_x + 1)
        x1 = torch.clamp(x1, zero, max_x + 1)

        x0_float = x0.float()
        x1_float = x1.float()
        y0_float = y0.float()
        y1_float = y1.float()

        vol_a0 = ((y1_float - y) * (x1_float - x)).unsqueeze(-1).to(self.device)
        vol_c0 = ((y1_float - y) * (x - x0_float)).unsqueeze(-1).to(self.device)
        vol_a1 = ((y - y0_float) * (x1_float - x)).unsqueeze(-1).to(self.device)
        vol_c1 = ((y - y0_float) * (x - x0_float)).unsqueeze(-1).to(self.device)

        outputs = (value_a0 * vol_a0 + value_c0 * vol_c0 + value_a1 * vol_a1 +
                   value_c1 * vol_c1)

        if self.morph == 0:
            outputs = outputs.reshape([
                self.num_batch,
                self.num_points * self.width,
                1 * self.height,
                self.num_channels,
            ])
            outputs = outputs.permute(0, 3, 1, 2)
        else:
            outputs = outputs.reshape([
                self.num_batch,
                1 * self.width,
                self.num_points * self.height,
                self.num_channels,
            ])
            outputs = outputs.permute(0, 3, 1, 2)
        return outputs

    def deform_conv(self, input, offset, if_offset):
        y, x = self._coordinate_map_3D(offset, if_offset)
        deformed_feature = self._bilinear_interpolate_3D(input, y, x)
        return deformed_feature

class EncoderConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(EncoderConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x


class DecoderConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(DecoderConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)

        return x


class DSCNet(nn.Module):

    def __init__(
        self,
        n_channels = 3,
        n_classes = 2,
        kernel_size = 9,
        extend_scope = 1.0,
        if_offset = True,
        device = 0,
        number = 16,
        dim = 1,
    ):
        """
        Our DSCNet
        :param n_channels: input channel
        :param n_classes: output channel
        :param kernel_size: the size of kernel
        :param extend_scope: the range to expand (default 1 for this method)
        :param if_offset: whether deformation is required, if it is False, it is the standard convolution kernel
        :param device: set on gpu
        :param number: basic layer numbers
        :param dim:
        """
        super(DSCNet, self).__init__()
        self.device = device
        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.if_offset = if_offset
        self.relu = nn.ReLU(inplace=True)
        self.number = number
        """
        The three contributions proposed in our paper are relatively independent. 
        In order to facilitate everyone to use them separately, 
        we first open source the network part of DSCNet. 
        <dim> is a parameter used by multiple templates, 
        which we will open source in the future ...
        """
        self.dim = dim  # This version dim is set to 1 by default, referring to a group of x-axes and y-axes
        """
        Here is our framework. Since the target also has non-tubular structure regions, 
        our designed model also incorporates the standard convolution kernel, 
        for fairness, we also add this operation to compare with other methods (like: Deformable Convolution).
        """
        self.conv00 = EncoderConv(n_channels, self.number)
        self.conv0x = DSConv(
            n_channels,
            self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            self.device,
        )
        self.conv0y = DSConv(
            n_channels,
            self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            self.device,
        )
        self.conv1 = EncoderConv(3 * self.number, self.number)

        self.conv20 = EncoderConv(self.number, 2 * self.number)
        self.conv2x = DSConv(
            self.number,
            2 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            self.device,
        )
        self.conv2y = DSConv(
            self.number,
            2 * self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            self.device,
        )
        self.conv3 = EncoderConv(6 * self.number, 2 * self.number)

        self.conv40 = EncoderConv(2 * self.number, 4 * self.number)
        self.conv4x = DSConv(
            2 * self.number,
            4 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            self.device,
        )
        self.conv4y = DSConv(
            2 * self.number,
            4 * self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            self.device,
        )
        self.conv5 = EncoderConv(12 * self.number, 4 * self.number)

        self.conv60 = EncoderConv(4 * self.number, 8 * self.number)
        self.conv6x = DSConv(
            4 * self.number,
            8 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            self.device,
        )
        self.conv6y = DSConv(
            4 * self.number,
            8 * self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            self.device,
        )
        self.conv7 = EncoderConv(24 * self.number, 8 * self.number)

        self.conv120 = EncoderConv(12 * self.number, 4 * self.number)
        self.conv12x = DSConv(
            12 * self.number,
            4 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            self.device,
        )
        self.conv12y = DSConv(
            12 * self.number,
            4 * self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            self.device,
        )
        self.conv13 = EncoderConv(12 * self.number, 4 * self.number)

        self.conv140 = DecoderConv(6 * self.number, 2 * self.number)
        self.conv14x = DSConv(
            6 * self.number,
            2 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            self.device,
        )
        self.conv14y = DSConv(
            6 * self.number,
            2 * self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            self.device,
        )
        self.conv15 = DecoderConv(6 * self.number, 2 * self.number)

        self.conv160 = DecoderConv(3 * self.number, self.number)
        self.conv16x = DSConv(
            3 * self.number,
            self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            self.device,
        )
        self.conv16y = DSConv(
            3 * self.number,
            self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            self.device,
        )
        self.conv17 = DecoderConv(3 * self.number, self.number)

        self.out_conv = nn.Conv2d(self.number, n_classes, 1)
        self.maxpooling = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2,
                              mode="bilinear",
                              align_corners=True)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # block0
        x_00_0 = self.conv00(x)
        x_0x_0 = self.conv0x(x)
        x_0y_0 = self.conv0y(x)
        x_0_1 = self.conv1(cat([x_00_0, x_0x_0, x_0y_0], dim=1))

        # block1
        x = self.maxpooling(x_0_1)
        x_20_0 = self.conv20(x)
        x_2x_0 = self.conv2x(x)
        x_2y_0 = self.conv2y(x)
        x_1_1 = self.conv3(cat([x_20_0, x_2x_0, x_2y_0], dim=1))

        # block2
        x = self.maxpooling(x_1_1)
        x_40_0 = self.conv40(x)
        x_4x_0 = self.conv4x(x)
        x_4y_0 = self.conv4y(x)
        x_2_1 = self.conv5(cat([x_40_0, x_4x_0, x_4y_0], dim=1))

        # block3
        x = self.maxpooling(x_2_1)
        x_60_0 = self.conv60(x)
        x_6x_0 = self.conv6x(x)
        x_6y_0 = self.conv6y(x)
        x_3_1 = self.conv7(cat([x_60_0, x_6x_0, x_6y_0], dim=1))

        # block4
        x = self.up(x_3_1)
        x_120_2 = self.conv120(cat([x, x_2_1], dim=1))
        x_12x_2 = self.conv12x(cat([x, x_2_1], dim=1))
        x_12y_2 = self.conv12y(cat([x, x_2_1], dim=1))
        x_2_3 = self.conv13(cat([x_120_2, x_12x_2, x_12y_2], dim=1))

        # block5
        x = self.up(x_2_3)
        x_140_2 = self.conv140(cat([x, x_1_1], dim=1))
        x_14x_2 = self.conv14x(cat([x, x_1_1], dim=1))
        x_14y_2 = self.conv14y(cat([x, x_1_1], dim=1))
        x_1_3 = self.conv15(cat([x_140_2, x_14x_2, x_14y_2], dim=1))

        # block6
        x = self.up(x_1_3)
        x_160_2 = self.conv160(cat([x, x_0_1], dim=1))
        x_16x_2 = self.conv16x(cat([x, x_0_1], dim=1))
        x_16y_2 = self.conv16y(cat([x, x_0_1], dim=1))
        x_0_3 = self.conv17(cat([x_160_2, x_16x_2, x_16y_2], dim=1))
        # x = self.dropout(x)
        out = self.out_conv(x_0_3)
        out = self.sigmoid(out)

        return out