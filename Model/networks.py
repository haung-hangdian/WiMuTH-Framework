import torch

import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler

from Model.aunetr2u import R2U_Net, WTU_Net, U_Net, AttU_Net, R2AttU_Net, AttU_Net_small, R2AttU_Net_smx, W_UNet, U_Net_Plus, AttU_Net_Plus, R2AttU_Net_Plus
from Model.TR import R2TRUnet,TRUnet
from Model.mynet import SCUNet, EGEUNet
from Model.hhynet import WTEGEU_Net, GCNU_Net,NEWU_Net
# from Model.ESPNet import EESPNet

from Model.fcn import FCN8
from Model.segnet import SegNet, SegResNet
from Model.enet import ENet
from Model.gcn import GCN
from Model.deeplabv3_plus import DeepLab, DeepLab_smx
# # from Model.duc_hdc import DeepLab_DUC_HDC
# # from Model.upernet import UperNet
from Model.pspnet import PSPNet
# from Model.pspnet import PSPDenseNet
from Model.unet import sk,skold, cbam
from Model.core.models.ocnet import OCNet
from Model.core.models.icnet import ICNet
from Model.core.models.lednet import LEDNet
from Model.core.models.dunet import DUNet
from Model.core.models.pspnet import PSPNet
from Model.core.models.bisenet import BiSeNet
from Model.core.models.psanet import PSANet
from Model.core.models.fcn import FCN8s
from Model.transunet_pytorch.utils.transunet import TransUNet
from Model.attunet import AttentionUNet


###############################################################################
# Imports
###############################################################################

# if opt.which_model_netG == 'resnet_9blocks':
#     from Model.aunet import R2U_Net, U_Net, AttU_Net, R2AttU_Net, AttU_Net_small
# elif opt.which_model_netG=='FCN8':
#     from Model.fcn import FCN8
# elif opt.which_model_netG=='SegNet':
#     from Model.segnet import SegNet
#     from Model.segnet import SegResNet
# elif opt.which_model_netG=='DeepLab':
#     from Model.deeplabv3_plus import DeepLab
#     from Model.duc_hdc import DeepLab_DUC_HDC
# elif opt.which_model_netG=='GCN':
#     from Model.gcn import GCN
# elif opt.which_model_netG=='PSPDenseNet':
#     from Model.pspnet import PSPNet
#     from Model.pspnet import PSPDenseNet
# elif opt.which_model_netG=='SegResNet':
#     netG = UnetGenerator3(input_nc, output_nc, which_model_netG ,gpu_ids=gpu_ids)
# elif opt.which_model_netG=='sk':
#     from Model.unet import sk
# elif opt.which_model_netG=='cbam':
#     from Model.unet import cbam

###############################################################################
# Functions
###############################################################################


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)

#     try:
    if classname.find('SeparableConv2d')!=-1:
        init.normal_(m.conv1.weight.data, 0.0, 0.02)
        init.normal_(m.pointwise.weight.data, 0.0, 0.02)
        init.normal_(m.bn.weight.data, 1.0, 0.02)
        init.constant(m.bn.bias.data, 0.0)

    elif classname.find('_ConvBNReLU') != -1:
        init.normal_(m.conv.weight.data, 0.0, 0.02)
        init.normal_(m.bn.weight.data, 1.0, 0.02)
        init.constant_(m.bn.bias.data, 0.0)
    elif classname.find('SKConv') != -1:
        #print(m.convs[0][0].weight)
        init.normal_(m.convs[0][0].weight.data, 0.0, 0.02)
        init.normal_(m.convs[0][1].weight.data, 1.0, 0.02)
        init.constant(m.convs[0][1].bias.data, 0.0)
        init.normal_(m.convs[1][0].weight.data, 0.0, 0.02)
        init.normal_(m.convs[1][1].weight.data, 1.0, 0.02)
        init.constant(m.convs[1][1].bias.data, 0.0)
        #raise Exception('TTT')
    elif classname.find('BasicConv') != -1:
        #print(m,m.conv)
        init.normal_(m.conv.weight.data, 0.0, 0.02)
        init.normal_(m.bn.weight.data, 1.0, 0.02)
        init.constant(m.bn.bias.data, 0.0)
    elif classname.find('ConvBlock') != -1:
        init.normal_(m.conv[0].weight.data, 0.0, 0.02)
        init.normal_(m.conv[1].weight.data, 1.0, 0.02)
        init.constant(m.conv[1].bias.data, 0.0)
        init.normal_(m.conv[3].weight.data, 0.0, 0.02)
        init.normal_(m.conv[4].weight.data, 1.0, 0.02)
        init.constant(m.conv[4].bias.data, 0.0)
    elif classname.find('UpConv') != -1:
        init.normal_(m.up[1].weight.data, 0.0, 0.02)
        init.normal_(m.up[2].weight.data, 1.0, 0.02)
        init.constant(m.up[2].bias.data, 0.0)
    elif classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)
    # elif classname.find('_ConvBNReLU') != -1:
    #     nn.init.normal_(m.weight.data, 0.0, 0.02)
        
        
#     except Exception as e:
#         print(e)
#         raise Exception('Error')
        #print(m.weight)
        
        

def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    #('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def get_norm_layer(norm_type='instance'):
    #print("normalization method [%s]" % norm_type)
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'fixed':
        def lambda_rule(epoch):
            if(opt.epoch_count == 1):
                lr_l = 0.0001
            elif(opt.epoch_count>1 and opt.epoch_count<6 ):
                lr_l = 0.00001
            else:    
                lr_l = 0.000000001 + (200-opt.epoch_count)*0.000000005
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal', gpu_ids=[],n_downsampling=2,num_classes=8,fineSize=512):
    netG = None
    use_gpu = len(gpu_ids) > 0

    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())


# from Model.enet import ENet
# from Model.upernet import UperNet




    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids,n_downsampling=n_downsampling)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids,n_downsampling=n_downsampling)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_9blocks_class':
        netG = ResnetGenerator_with_Classifier(input_nc, output_nc, ngf, num_classes=num_classes, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids,n_downsampling=n_downsampling)
    elif which_model_netG in ['R2U_Net','NEWU_Net','GCNU_Net','WTEGEU_Net','EGEUNet','EGEUNet','WTU_Net','SCUNet','R2AttU_Net_Plus','AttU_Net_Plus', 'U_Net_Plus','U_Net', 'W_UNet', 'AttU_Net', 'R2AttU_Net', 'AttU_Net_small', 'R2TRUnet','TRUnet', 'R2AttU_Net_smx']:
        netG = UnetGenerator2(input_nc, output_nc,which_model_netG,gpu_ids=gpu_ids)
    elif which_model_netG=='EESPNet':
        netG = UnetGenerator4(classes = input_nc)
    elif which_model_netG=='FCN8':
        netG = UnetGenerator3(input_nc, output_nc, which_model_netG ,gpu_ids=gpu_ids)
    elif which_model_netG=='OCNet':
        netG = UnetGeneratorOCNet(input_nc, output_nc, which_model_netG ,gpu_ids=gpu_ids)
    elif which_model_netG=='ICNet':
        netG = UnetGeneratorICNet(input_nc, output_nc, which_model_netG ,gpu_ids=gpu_ids)
    elif which_model_netG=='LEDNet':
        netG = UnetGeneratorLEDNet(input_nc, output_nc, which_model_netG ,gpu_ids=gpu_ids)
    elif which_model_netG=='DUNet':
        netG = UnetGeneratorDUNet(input_nc, output_nc, which_model_netG ,gpu_ids=gpu_ids)
    elif which_model_netG=='PSPNet':
        netG = UnetGeneratorPSPNet(input_nc, output_nc, which_model_netG ,gpu_ids=gpu_ids)
    elif which_model_netG=='BiSeNet':
        netG = UnetGeneratorBiSeNet(input_nc, output_nc, which_model_netG ,gpu_ids=gpu_ids)
    elif which_model_netG=='PSANet':
        print('fineSize = ' , fineSize)
        netG = UnetGeneratorPSANet(input_nc, output_nc, which_model_netG ,gpu_ids=gpu_ids)
    elif which_model_netG=='SegNet':
        netG = UnetGenerator3(input_nc, output_nc, which_model_netG ,gpu_ids=gpu_ids)
    elif which_model_netG in ['DeepLab','DeepLab_smx']:
        netG = UnetGenerator3(input_nc, output_nc, which_model_netG ,gpu_ids=gpu_ids)
    elif which_model_netG=='GCN':
        netG = UnetGenerator3(input_nc, output_nc, which_model_netG ,gpu_ids=gpu_ids)
    elif which_model_netG=='FCN8s':
        netG = UnetGeneratorFCN8s(input_nc, output_nc, which_model_netG ,gpu_ids=gpu_ids)
    elif which_model_netG=='TransUNet':
        netG = UnetGeneratorTransUNet(input_nc, output_nc, which_model_netG ,gpu_ids=gpu_ids)
    elif which_model_netG in ['GCN','PSPNet', 'PSPDenseNet', 'ENet', 'SegNet','PSPDenseNet','SegResNet']:
        netG = UnetGenerator3(input_nc, output_nc, which_model_netG ,gpu_ids=gpu_ids)
    elif which_model_netG in ['sk','skold']:
#         netG = sk(img_ch=input_nc, n_classes=output_nc, reduction_ratio=None, conv_type = which_model_netG)
        netG = UnetGenerator4(input_nc, output_nc, which_model_netG ,gpu_ids=gpu_ids)
    elif which_model_netG=='cbam':
#         netG = cbam(img_ch=input_nc, n_classes=output_nc, att_mode = which_model_netG)
        netG = UnetGenerator4(input_nc, output_nc, which_model_netG ,gpu_ids=gpu_ids)
    elif which_model_netG=='AttentionUNet':
        netG = attUnetGenerator(input_nc, output_nc, which_model_netG ,gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    init_weights(netG, init_type=init_type)
    return netG

class UnetGenerator4(nn.Module):
    def __init__(self, classes,s, gpu_ids=[]):
        super(UnetGenerator4, self).__init__()
        #R2U_Net, U_Net, AttU_Net, R2AttU_Net
        self.model = eval(which_model_netG)(classes = output_nc)
    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

        

class attUnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc,which_model_netG, use_dropout=False, gpu_ids=[]):
        super(attUnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids
        #R2U_Net, U_Net, AttU_Net, R2AttU_Net
        self.model = eval(which_model_netG)(img_ch=input_nc, output_ch=output_nc)
    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)
                

class UnetGenerator3(nn.Module):
    def __init__(self, input_nc, output_nc,which_model_netG, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator3, self).__init__()
        self.gpu_ids = gpu_ids
        #R2U_Net, U_Net, AttU_Net, R2AttU_Net
        self.model = eval(which_model_netG)(num_classes = output_nc, in_channels=input_nc)
    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)
        
class UnetGeneratorICNet(nn.Module):
    def __init__(self, input_nc, output_nc,which_model_netG, use_dropout=False, gpu_ids=[]):
        super(UnetGeneratorICNet, self).__init__()
        self.gpu_ids = gpu_ids
        #R2U_Net, U_Net, AttU_Net, R2AttU_Net
        self.model = eval(which_model_netG)(nclass = output_nc, pretrained_base = False)
    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)
        
class UnetGeneratorOCNet(nn.Module):
    def __init__(self, input_nc, output_nc,which_model_netG, use_dropout=False, gpu_ids=[]):
        super(UnetGeneratorOCNet, self).__init__()
        self.gpu_ids = gpu_ids
        #R2U_Net, U_Net, AttU_Net, R2AttU_Net
        self.model = eval(which_model_netG)(nclass = output_nc, pretrained_base = False)
    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)
class UnetGeneratorLEDNet(nn.Module):
    def __init__(self, input_nc, output_nc,which_model_netG, use_dropout=False, gpu_ids=[]):
        super(UnetGeneratorLEDNet, self).__init__()
        self.gpu_ids = gpu_ids
        #R2U_Net, U_Net, AttU_Net, R2AttU_Net
        self.model = eval(which_model_netG)(nclass = output_nc, pretrained_base = False)
    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)
        
class UnetGeneratorDUNet(nn.Module):
    def __init__(self, input_nc, output_nc,which_model_netG, use_dropout=False, gpu_ids=[]):
        super(UnetGeneratorDUNet, self).__init__()
        self.gpu_ids = gpu_ids
        #R2U_Net, U_Net, AttU_Net, R2AttU_Net
        self.model = eval(which_model_netG)(nclass = output_nc, pretrained_base = False)
    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)
class UnetGeneratorPSPNet(nn.Module):
    def __init__(self, input_nc, output_nc,which_model_netG, use_dropout=False, gpu_ids=[]):
        super(UnetGeneratorPSPNet, self).__init__()
        self.gpu_ids = gpu_ids
        #R2U_Net, U_Net, AttU_Net, R2AttU_Net
        self.model = eval(which_model_netG)(nclass = output_nc, pretrained_base = False)
    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)
class UnetGeneratorBiSeNet(nn.Module):
    def __init__(self, input_nc, output_nc,which_model_netG, use_dropout=False, gpu_ids=[]):
        super(UnetGeneratorBiSeNet, self).__init__()
        self.gpu_ids = gpu_ids
        #R2U_Net, U_Net, AttU_Net, R2AttU_Net
        self.model = eval(which_model_netG)(nclass = output_nc, pretrained_base = False)
    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)
class UnetGeneratorPSANet(nn.Module):
    def __init__(self, input_nc, output_nc, which_model_netG, use_dropout=False, gpu_ids=[]):
        super(UnetGeneratorPSANet, self).__init__()
        self.gpu_ids = gpu_ids
        #R2U_Net, U_Net, AttU_Net, R2AttU_Net
        self.model = eval(which_model_netG)(nclass = output_nc,pretrained_base = False)
    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)
        
        
             
class UnetGeneratorFCN8s(nn.Module):
    def __init__(self, input_nc, output_nc, which_model_netG, use_dropout=False, gpu_ids=[]):
        super(UnetGeneratorFCN8s, self).__init__()
        self.gpu_ids = gpu_ids
        self.model = eval(which_model_netG)(nclass = output_nc,in_channels=input_nc)
    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)
        
        
class UnetGeneratorTransUNet(nn.Module):
    def __init__(self, input_nc, output_nc, which_model_netG, use_dropout=False, gpu_ids=[]):
        super(UnetGeneratorTransUNet, self).__init__()
        self.gpu_ids = gpu_ids
        self.model = eval(which_model_netG)(img_dim=512,in_channels=input_nc,out_channels=128,head_num=4,mlp_dim=512,block_num=8,patch_dim=16,class_num=output_nc)
   ##img_dim如果处理别的尺寸的数据集也许需要修改
    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)
        
        

class UnetGenerator4(nn.Module):
    def __init__(self, input_nc, output_nc,which_model_netG, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator4, self).__init__()
        self.gpu_ids = gpu_ids
        #R2U_Net, U_Net, AttU_Net, R2AttU_Net
        self.model = eval(which_model_netG)(img_ch=input_nc, n_classes=output_nc, reduction_ratio=None)
    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[],num_classes=8):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
        
    elif which_model_netD == 'AC':
        netD = ACDiscriminator(input_nc, ndf, num_classes=num_classes ,n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids,num_classes=num_classes)
    elif which_model_netD == 'pixel':
        netD = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    if use_gpu:
        netD.cuda(gpu_ids[0])
    init_weights(netD, init_type=init_type)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################

##Add a loss for two classify vector from two different kinds of images

class Class_Idt_Loss_l1(torch.nn.Module):
    def __init__(self):
        super(Class_Idt_Loss_l1,self).__init__()
        
    def forward(self,x,y):
        totloss = torch.abs(x-y)
        totloss = torch.sum(totloss)
        return totloss

class Class_Idt_Loss(torch.nn.Module):
    def __init__(self):
        super(Class_Idt_Loss,self).__init__()
        
    def forward(self,x,y):
        #totloss = torch.abs(x-y)
        totloss  = torch.nn.functional.pairwise_distance(x,y)
        totloss = torch.sum(totloss)
        return totloss


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        #tmp_targets = targets.view(-1,1,1)
        tmp_targets = targets.view(-1,inputs.shape[-2],inputs.shape[-1])
        return self.nll_loss(torch.nn.functional.log_softmax(inputs), tmp_targets)


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, gan_mode='vanilla', target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
            
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect',n_downsampling=2):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        #********
        #  John Changed
        #********
        #n_downsampling = 2
        #n_downsampling = 3    

        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

#
# John Added
# Add a classifier layer 
#
class ResnetGenerator_with_Classifier(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, num_classes=8, \
                    norm_layer=nn.BatchNorm2d, use_dropout=False,\
                     n_blocks=6, gpu_ids=[], padding_type='reflect',\
                     n_downsampling=2):
        assert(n_blocks >= 0)
        super(ResnetGenerator_with_Classifier, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]  

        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        


        mult = 2**n_downsampling
        for i in range(n_blocks):
            model +=  [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        model_org = []
        model_cl = []

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            if i == 0:
                model_org += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                                        norm_layer(int(ngf * mult / 2)),
                                        nn.ReLU(True)]
            else:
                model_org += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]

        model_org += [nn.ReflectionPad2d(3)]
        model_org += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model_org += [nn.Tanh()]

        self.model_base = nn.Sequential(*model)
        self.model_org = nn.Sequential(*model_org)
        self.model_cl =  Classifer(num_classes)


    def forward(self, input):

        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            x = nn.parallel.data_parallel(self.model_base,input,self.gpu_ids)
            a = nn.parallel.data_parallel(self.model_org,x,self.gpu_ids)
            b = nn.parallel.data_parallel(self.model_cl,x,self.gpu_ids)
            return a,b
        else:
            x = self.model_base(input)
            return self.model_org(x),self.model_cl(x)

class Classifer_legacy(nn.Module):
    def __init__(self,num_classes=8):
        super(Classifer,self).__init__()

        self.conv_block = nn.Sequential(nn.Conv2d(256,64, kernel_size=7,stride=2, padding=0,),\
                                            nn.InstanceNorm2d(64),\
                                            nn.ReLU(True),\
                                            nn.Conv2d(64,32, kernel_size=5,stride=2, padding=0,),\
                                            nn.InstanceNorm2d(64),\
                                            nn.ReLU(True),\
                                            nn.Conv2d(32,32, kernel_size=3,stride=2, padding=0,))
        self.fc = nn.Sequential(nn.Linear(32*6*6,num_classes),\
                                nn.Sigmoid(),)
    def forward(self,x):
        x = self.conv_block(x)
        x = x.view(-1,32*6*6)
        return self.fc(x)

class Classifer(nn.Module):
    def __init__(self,num_classes=8):
        super(Classifer,self).__init__()

        self.conv_block = nn.Sequential(nn.Conv2d(256,64, kernel_size=3,stride=2, padding=0,),\
                                   nn.InstanceNorm2d(64),\
                                   nn.MaxPool2d(kernel_size=2),\
                                   nn.ReLU(True),\
                                   nn.Conv2d(64,32, kernel_size=3,stride=2, padding=0,),\
                                   nn.InstanceNorm2d(32),\
                                   nn.MaxPool2d(kernel_size=2),\
                                   nn.ReLU(True),\
                                   nn.Conv2d(32,num_classes, kernel_size=3,stride=3, padding=0,))


    def forward(self,x):
        return self.conv_block(x)



# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, iput_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


class UnetGenerator2(nn.Module):
    def __init__(self, input_nc, output_nc,which_model_netG, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator2, self).__init__()
        self.gpu_ids = gpu_ids
        #R2U_Net, U_Net, AttU_Net, R2AttU_Net
        self.model = eval(which_model_netG)(input_nc, output_nc)
    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)        


# class MyNet(nn.Module):
#     def __init__(self, inp_dim, mod_dim1, mod_dim2):
#         super(MyNet, self).__init__()

#         self.seq = nn.Sequential(
#             nn.Conv2d(inp_dim, mod_dim1, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(mod_dim1),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(mod_dim1, mod_dim2, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(mod_dim2),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(mod_dim2, mod_dim1, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(mod_dim1),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(mod_dim1, mod_dim2, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(mod_dim2),
#         )

#     def forward(self, x):
#         return self.seq(x)

# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[],num_classes=8):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(PixelDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.net, input, self.gpu_ids)
        else:
            return self.net(input)

import torch
from torch import nn
import functools

# Defines the PatchGAN discriminator with the specified arguments.
class ACDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3,norm_layer=nn.BatchNorm2d,\
        use_sigmoid=False, gpu_ids=[], num_classes= 8):
        super(ACDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence_rf  = sequence + [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        
        sequence_cl  = sequence +\
                        [nn.Conv2d(ndf * nf_mult, ndf * nf_mult,
                                  kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                        norm_layer(ndf * nf_mult),
                        nn.LeakyReLU(0.2, True),
                        #nn.MaxPool2d(kernel_size=2,padding=padw),
                        ]+\
                        [nn.Conv2d(ndf * nf_mult, ndf * nf_mult,
                        kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                        norm_layer(ndf * nf_mult),
                        nn.LeakyReLU(0.2, True),
                        nn.MaxPool2d(kernel_size=2,padding=padw),
                        ]+\
                        [nn.Conv2d(ndf * nf_mult, num_classes, kernel_size=4, stride=1, padding=0)]
        
        if use_sigmoid:
            sequence_rf += [nn.Sigmoid()]
            sequence_cl += [nn.Sigmoid()]
            

        self.model_rf = nn.Sequential(*sequence_rf)
        self.model_cl = nn.Sequential(*sequence_cl)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model_rf, input, self.gpu_ids),nn.parallel.data_parallel(self.model_cl, input, self.gpu_ids)
        else:
            return self.model_rf(input),self.model_cl(input)