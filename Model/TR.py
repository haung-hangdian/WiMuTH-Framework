import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
###############################################################################
# Functions
###############################################################################
from Model.swintr import SwinTransformer as swin

class UnetBlock_(nn.Module):
    def __init__(self, up_in1,up_in2,up_out):
        super().__init__()

        self.x_conv = nn.Conv2d(up_in1, up_out, kernel_size=3, padding=1)
        self.x_conv_ = nn.Conv2d(up_in2, up_in1, kernel_size=1, padding=0)

        self.bn = nn.BatchNorm2d(up_out)


        # self.deconv = nn.ConvTranspose2d(2208, 2208, 3, stride=2, padding=1, output_padding=1)
        # nn.init.xavier_normal_(self.deconv.weight)

        #  init my layers
        nn.init.xavier_normal_(self.x_conv.weight)
        nn.init.xavier_normal_(self.x_conv_.weight)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, up_p, x_p):

        up_p = F.interpolate(up_p, scale_factor=2, mode='bilinear', align_corners=True)
        # up_p = self.deconv(up_p)

        x_p = self.x_conv_(x_p)
        cat_p = torch.add(up_p, x_p)
        cat_p = self.x_conv(cat_p)
        cat_p = F.relu(self.bn(cat_p))

        return cat_p

class conv_batch(nn.Module):
    def __init__(self,patch_size,num_head,height,width,num_layer,in_channel,out_channel):
        super().__init__()
        self.height=height
        self.width=width
        self.num_layer=num_layer
        self.patch_size =patch_size
        self.base_channel =num_head
        
        self.in_channel = in_channel
        self.out_channel =out_channel
        
        self.conv = nn.Conv2d(in_channel,out_channel,kernel_size=3,padding=1)
        self.bn = nn.BatchNorm2d(out_channel)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1)
        elif isinstance(m,nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
        
    def forward (self,x,feature):
        b,c,h,w = x.shape
#         print('conv-',feature.shape)
        feature = feature.contiguous().reshape(b,self.base_channel*(2**(self.num_layer)),self.height//(self.patch_size*(2**(self.num_layer))),self.width//(self.patch_size*(2**(self.num_layer))))
#         print('conv1-',feature.shape)
        feature = F.interpolate(feature, scale_factor=(self.patch_size*(2**(self.num_layer))), mode='bilinear', align_corners=True)
        
        feature = self.conv(feature)
        feature = F.relu(self.bn(feature))
        
        x = torch.mul(x,feature)
        
        return x
    

class conv_batch_last(nn.Module):
    def __init__(self,patch_size,num_head,height,width,out_channel,idx):
        super().__init__()
        self.height=height
        self.width=width
        self.idx =idx
        self.patch_size= patch_size
        
        
        self.in_channel =num_head*idx
        self.out_channel =out_channel
        
        self.conv = nn.Conv2d(self.in_channel,out_channel,kernel_size=3,padding=1)
        self.bn = nn.BatchNorm2d(out_channel)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1)
        elif isinstance(m,nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
        
    def forward (self,x,feature):
        b,c,h,w = x.shape
#         print('nnconv-',feature.shape)
        feature = feature.contiguous().reshape(b,self.in_channel,self.height//(self.patch_size*self.idx),self.width//(self.patch_size*self.idx))
#         print('nnconv1-',feature.shape)
        feature = F.interpolate(feature, scale_factor=self.patch_size*self.idx, mode='bilinear', align_corners=True)
        
        feature = self.conv(feature)
        feature = F.relu(self.bn(feature))
        
        x = torch.mul(x,feature)
        
        return x
        
class encoder_feature(nn.Module):
    def __init__(self,num_head,patch_size,out_channel,num_layers,height,width,idx):
        
        super().__init__()

        self.out_channel = out_channel
        self.num_heads =num_head
        self.num_layers=num_layers
        self.idx =idx
        
        self.layers =nn.ModuleList()
        
        for num_layer in range (1,num_layers):
            
            if num_layer < num_layers-2:
                
                layer = conv_batch(patch_size,num_head,height,width,num_layer,self.num_heads*(2**(num_layer)),out_channel)
            
            else:
                layer = conv_batch_last(patch_size,num_head,height,width,out_channel,self.idx)
                
            self.layers.append(layer)
        
    
    def forward(self,x,feature):
        b,c,h,w = x.shape # x is the cnn feature
        encode_feature=[]
        index =1
        for layer in self.layers:
            
            if index <self.num_layers:
        
                en_feature = layer(x,feature[index])
                encode_feature.append(en_feature)
                index +=1
            
        for i in range(len(encode_feature)):
            
            x = x + encode_feature[i]
        
        return x

def init_weights(net, init_type='normal', gain=0.02):
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

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
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

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Recurrent_block(nn.Module):
    def __init__(self,ch_out,t=2):
        super(Recurrent_block,self).__init__()
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
        
class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1


class single_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(single_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class TRUnet(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(TRUnet,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=32)
        self.Conv2 = conv_block(ch_in=32,ch_out=64)
        self.Conv3 = conv_block(ch_in=64,ch_out=128)
        self.Conv4 = conv_block(ch_in=128,ch_out=256)

        self.Up4 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)
        
        self.Up3 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)
        
        self.Up2 = up_conv(ch_in=64,ch_out=32)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)
        
        self.swin4 = swin(patch_size=4,in_chans=256,depths=(2,), num_heads=(3,))
        self.swin3 = swin(patch_size=4,in_chans=128,depths=(2,2), num_heads=(3,6))
        self.swin2 = swin(patch_size=4,in_chans=64,depths=(2,2,6), num_heads=(3,6,12))
        self.swin1 = swin(patch_size=4,in_chans=32,depths=(2,2,6,2), num_heads=(3,6,12,24))
        
        self.enf4 = encoder_feature(patch_size=4,num_head=96,out_channel=256,num_layers = 2,height=32,width=32,idx=1) 
        self.enf3 = encoder_feature(patch_size=4,num_head=96,out_channel=128,num_layers = 3,height=64,width=64,idx=2)
        self.enf2 = encoder_feature(patch_size=4,num_head=96,out_channel=64,num_layers = 4,height=128,width=128,idx=4)
        self.enf1 = encoder_feature(patch_size=4,num_head=96,out_channel=32,num_layers = 5,height=256,width=256,idx=8)
        
#         self.up = UnetBlock_(512,256,256)
    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x) #32*256*256
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2) #64*128*128
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3) #128*64*64


        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4) #256*32*32

        x4_feature = self.swin4(x4)
        encoded_x4 = self.enf4(x4,x4_feature)
        
        d4 = self.Up4(encoded_x4)
        x3_feature = self.swin3(x3)
        encoded_x3 = self.enf3(x3,x3_feature)
        d4 = torch.cat((encoded_x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2_feature = self.swin2(x2)
        encoded_x2 =self.enf2(x2,x2_feature)
        d3 = torch.cat((encoded_x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1_feature = self.swin1(x1)
        encoded_x1 = self.enf1(x1,x1_feature)
        d2 = torch.cat((encoded_x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class R2TRUnet(nn.Module):
    def __init__(self,img_ch=3,output_ch=1,t=2):
        super(R2TRUnet,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=32,t=t)

        self.RRCNN2 = RRCNN_block(ch_in=32,ch_out=64,t=t)
        
        self.RRCNN3 = RRCNN_block(ch_in=64,ch_out=128,t=t)
        
        self.RRCNN4 = RRCNN_block(ch_in=128,ch_out=256,t=t)
        
        self.Up4 = up_conv(ch_in=256,ch_out=128)
        self.Up_RRCNN4 = RRCNN_block(ch_in=256, ch_out=128,t=t)
        
        self.Up3 = up_conv(ch_in=128,ch_out=64)
        self.Up_RRCNN3 = RRCNN_block(ch_in=128, ch_out=64,t=t)
        
        self.Up2 = up_conv(ch_in=64,ch_out=32)
        self.Up_RRCNN2 = RRCNN_block(ch_in=64, ch_out=32,t=t)

        self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)
        
        self.swin4 = swin(patch_size=4,in_chans=256,depths=(2,), num_heads=(3,))
        self.swin3 = swin(patch_size=4,in_chans=128,depths=(2,2), num_heads=(3,6))
        self.swin2 = swin(patch_size=4,in_chans=64,depths=(2,2,6), num_heads=(3,6,12))
        self.swin1 = swin(patch_size=4,in_chans=32,depths=(2,2,6,2), num_heads=(3,6,12,24))
        
        self.enf4 = encoder_feature(patch_size=4,num_head=96,out_channel=256,num_layers = 2,height=32,width=32,idx=1) 
        self.enf3 = encoder_feature(patch_size=4,num_head=96,out_channel=128,num_layers = 3,height=64,width=64,idx=2)
        self.enf2 = encoder_feature(patch_size=4,num_head=96,out_channel=64,num_layers = 4,height=128,width=128,idx=4)
        self.enf1 = encoder_feature(patch_size=4,num_head=96,out_channel=32,num_layers = 5,height=256,width=256,idx=8)
        self.softmax = nn.Softmax(dim=1)   # 添加softmax层


    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        # decoding + concat path
        
        x4_feature = self.swin4(x4)
        encoded_x4 = self.enf4(x4,x4_feature)
        
        d4 = self.Up4(encoded_x4)
        x3_feature = self.swin3(x3)
        encoded_x3 = self.enf3(x3,x3_feature)
        d4 = torch.cat((encoded_x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2_feature = self.swin2(x2)
        encoded_x2 =self.enf2(x2,x2_feature)
        d3 = torch.cat((encoded_x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1_feature = self.swin1(x1)
        encoded_x1 = self.enf1(x1,x1_feature)
        d2 = torch.cat((encoded_x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)
        return self.softmax(d1)     # 添加softmax操作

        return d1
    
