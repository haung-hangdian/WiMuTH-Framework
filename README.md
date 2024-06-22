# WiMuTH-Framework
This repo implements the segmentation method in the paper WiMuTH: a trustworthy Wavelet-Integrated Multidimensional Tensorial-Harmonic Coronary Artery Reconstruction Framework.

You are my ![Visitor Count](https://profile-counter.glitch.me/hauang-hangdian/count.svg) visitor, Thank You! &#x1F618;&#x1F618;

![](https://github.com/haung-hangdian/WiMuTH-Framework/blob/main/figures/overview.png)

<p align="center">Fig 1. Detailed framework structure of the WiMuTH.</p>


The WiMuTH Framework is a framework for Lumen and EEM boundary segmentation for IVUS images with contrast, artifacts, and fuzzy regions. It features four innovative modules: WCSR, MaHS, IDC, and DFCH bridge. WiMuTH Framework achieves state-of-the-art performance over 13 previous methods on the NIRS-IVUS datasets.


First, the methodology and underlying principles will be explained. Then, the comparative methods' experimental environment and the GitHub repositories will be delineated. Finally, the experimental outcomes will be presented.

## Method
### WCSR Module

WCSR leverages wavelet transformation to analyze the frequency domain characteristics of images, thereby enhancing model robustness against motion artifacts induced by cardiac pulsations and improving the capture of spatial information and boundary feature recognition.

### MaHS Module

![](https://github.com/haung-hangdian/WiMuTH-Framework/blob/main/figures/modules.png)

<p align="center">Fig 2. The realization of three different kinds of attention in MaHS module.</p>

MaHS integrates various attention mechanisms and convolution operations to enhance the model's capability of extracting and analyzing complex vascular region features.

### IDC Module

IDC employs deep normalization and adjustment of feature maps to optimize the integrity of information during feature transfer. This minimizes information loss and thus enhances the precision and consistency of segmentation.

### DFCH bridge

DFCH bridge enhances the restoration of spatial information during the feature upsampling process by fusing multi-layer features and applying convolutions with varied dilation rates. Thus, it augments the model’s ability to capture image details, particularly at image edges and boundaries.

### MAE-Enhanced Vascular Morphology Analysis

## Installation

We run WiMuTH Framework and previous methods on a system running Ubuntu 22.04, with Python 3.9, PyTorch 2.0.0, and CUDA 11.7. 

## Experiment

### Baselines
We provide GitHub links pointing to the PyTorch implementation code for all networks compared in this experiment here, so you can easily reproduce all these projects.

[PSPNet](https://github.com/hszhao/PSPNet); [OCNet](https://github.com/openseg-group/OCNet.pytorch); [TransUNet](https://github.com/Beckschen/TransUNet); [Attention UNet](https://github.com/pecheb/Att-Net); [UNet](https://github.com/milesial/Pytorch-UNet); [FCN-8s](https://github.com/pierluigiferrari/fcn8s_tensorflow); [ENet](https://github.com/TimoSaemann/ENet); [LEDNet](https://github.com/xiaoyufenfei/LEDNet); [GCN](https://github.com/SConsul/Global_Convolutional_Network); [DeepLab](https://github.com/fregu856/deeplabv3); [BiseNet](https://github.com/CoinCheung/BiSeNet) 
### Compare with others on the NIRS-IVUS dataset

![](https://github.com/haung-hangdian/WiMuTH-Framework/blob/main/tables/Baselines.png)

<p align="center">Fig 3. Comparison experiments between our method and 11 previous segmentation methods on the NIRS-IVUS dataset.</p>

![](https://github.com/haung-hangdian/WiMuTH-Framework/blob/main/figures/Calcification.png)

<p align="center">Fig 4. Our method was compared with 11 existing methods to visualize the calcification segmentation in the NIRS-IVUS dataset.</p>

![](https://github.com/haung-hangdian/WiMuTH-Framework/blob/main/figures/Side_branch.png)

<p align="center">Fig 4. Our method was compared with 11 existing methods to visualize the segmentation of side branches in the NIRS-IVUS dataset.</p>

![](https://github.com/haung-hangdian/WiMuTH-Framework/blob/main/figures/Dilated_vesse.png)

<p align="center">Fig 4. Our method was compared with 11 existing methods to visualize the segmentation of dilated vessels in the NIRS-IVUS dataset.</p>

![](https://github.com/haung-hangdian/WiMuTH-Framework/blob/main/figures/Constricted_vessel.png)

<p align="center">Fig 4. Our method was compared with 11 existing methods to visualize the segmentation of constrivted vessels in the NIRS-IVUS dataset.</p>

### Ablation study
#### Key components of WiMuTH
#### Extension of the number of MaHS
#### Hyperparameter of WCSR's order


