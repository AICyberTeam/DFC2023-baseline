import torch
import torch.nn as nn
from torch.nn import functional as F

affine_par = True

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(planes),
                                   nn.ReLU(inplace=False))
        self.conv2 = nn.Sequential(nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation*multi_grid, dilation=dilation*multi_grid, bias=False),
                                   nn.BatchNorm2d(planes),
                                   nn.ReLU(inplace=False))
        self.conv3 = nn.Sequential(nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(planes * 4))

        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual      
        out = self.relu_inplace(out)
        return out

class ASPPModule(nn.Module):
    """
    Reference: 
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """
    def __init__(self, features, inner_features=256, out_features=512, dilations=(12, 24, 36)):
        super(ASPPModule, self).__init__()

        self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                   nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
                                   nn.BatchNorm2d(inner_features))
        self.conv2 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
                                   nn.BatchNorm2d(inner_features))
        self.conv3 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
                                   nn.BatchNorm2d(inner_features))
        self.conv4 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
                                   nn.BatchNorm2d(inner_features))
        self.conv5 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
                                   nn.BatchNorm2d(inner_features))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(inner_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(out_features),
            nn.Dropout2d(0.1)
            )
        
    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.interpolate(self.conv1(x), size=(h, w), mode='bilinear', align_corners=True)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)

        bottle = self.bottleneck(out)
        return bottle

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(conv3x3(3, 64, stride=2),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=False))
        self.conv2 = nn.Sequential(conv3x3(64, 64),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=False))
        self.conv3 = nn.Sequential(conv3x3(64, 128),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(inplace=False))

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.relu = nn.ReLU(inplace=False)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, multi_grid=(1,1,1))

        # self.head = nn.Sequential(ASPPModule(2048),
        #                           nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=True),
        #                           nn.Sigmoid())

        self.head = nn.Sequential(ASPPModule(2048),
                                  nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(512),
                                  nn.Dropout2d(0.1),
                                  nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=True),
                                  nn.Sigmoid())

        self.dsn = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
                                 nn.BatchNorm2d(512),
                                 nn.Dropout2d(0.1),
                                 nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_dsn = self.dsn(x)
        x = self.layer4(x)

        x = self.head(x)

        return [x, x_dsn]

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index%len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample, multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)


def Res_Deeplab(num_classes=6):
    model = ResNet(Bottleneck,[3, 4, 23, 3], num_classes)
    return model

