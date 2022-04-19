# Modified ResNet with Conditional Batch Norm (CBN) layers instead of the batch norm layers
# Features from block 4 are used for the VQA task

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch

from cbn import CBN
import pdb

from sequential_modified import Sequential

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
'''
This modules returns both the conv feature map and the lstm question embedding (unchanges)
since subsequent CBN layers in nn.Sequential will require both inputs
'''
class Conv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, bias=True):
        super(Conv2d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, bias=bias)

    def forward(self, x, attributes):
        out = self.conv(x)
        return out, attributes


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, attributes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = CBN(planes, attributes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = CBN(planes, attributes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, attributes):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out, attributes)
        out = self.relu(out)

        out = self.conv2(out)
        out, _ = self.bn2(out, attributes)

        if self.downsample is not None:
            residual, _ = self.downsample(x, attributes)

        out += residual
        out = self.relu(out)

        return out, attributes


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, attributes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = CBN(planes, attributes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = CBN(planes, attributes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(planes * 4)
        self.bn3 = CBN(planes * 64, attributes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, attributes):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out, attributes)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, attributes)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out, attributes)

        if self.downsample is not None:
            residual, _ = self.downsample(x, attributes)

        out += residual
        out = self.relu(out)

        return out, attributes

class ResNet(nn.Module):

    def __init__(self, block, layers, attributes, num_classes=1000):
        self.inplanes = 64
        self.attributes = attributes
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False).to(DEVICE)
        # self.bn1 = nn.BatchNorm2d(64)
        # pdb.set_trace()
        self.bn1 = CBN(64, attributes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # pdb.set_trace()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                CBN(planes * block.expansion, self.attributes),
                # nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.attributes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.attributes))

        return Sequential(*layers)

    def forward(self, x, attributes):
        x = self.conv1(x)
        x = self.bn1(x, attributes)
        x = self.relu(x)
        x = self.maxpool(x)

        x, _ = self.layer1(x, attributes)
        x, _ = self.layer2(x, attributes)
        x, _ = self.layer3(x, attributes)
        x, _ = self.layer4(x, attributes)
        # pdb.set_trace()
        # not required currently
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x

def resnet18(attributes, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], attributes, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model


def resnet34(attributes,pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3],attributes,**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
    return model


def resnet50(attributes, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], attributes, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model


def resnet101(attributes, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], attributes, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
    return model


def resnet152(attributes, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], attributes, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']), strict=False)
    return model