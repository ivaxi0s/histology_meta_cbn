import torch.nn as nn
import torch
from cbn import CBN

from sequential_modified import Sequential


__all__ = ['ResNet', 'resnet18', 'resnet20', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

class BatchNorm2d(torch.nn.BatchNorm2d):
    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.fill_(1.0)
            self.bias.data.zero_()

''' Resnet '''

def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, attributes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None, remove_last_relu=False):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = CBN(planes, attributes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = CBN(planes, attributes)
        self.downsample = downsample
        self.stride = stride
        self.remove_last_relu = False

    def forward(self, x, attributes):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out, attributes)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, attributes)

        if self.downsample is not None:
            identity = self.downsample(x, attributes)

        out += identity

        if self.remove_last_relu:
            out = out
        else:
            out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, attributes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None, remove_last_relu=False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = CBN(width, attributes)
        self.conv2 = conv3x3(width, width, stride, groups)
        self.bn2 = CBN(width, attributes)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = CBN(planes * self.expansion, attributes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.remove_last_relu = remove_last_relu

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out, attributes)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, attributes)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out, attributes)

        if self.downsample is not None:
            identity = self.downsample(x, attributes)

        out += identity

        if self.remove_last_relu:
            out = out
        else:
            out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block_type, layers, attributes, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None, 
                 remove_last_relu=False, input_high_res=True):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        assert block_type in ['basic', 'bottleneck']

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.attributes = attributes

        if not input_high_res:
            self.layer0 = Sequential(
                nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                                     bias=False), 
                CBN(self.inplanes, self.attributes),
                nn.ReLU(inplace=True)
            )
        else:
            self.layer0 = Sequential(
                nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                          bias=False),
                CBN(self.inplanes, self.attributes),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )

        
        
        self.layer1 = self._make_layer(
            block_type, 64, layers[0], norm_layer=norm_layer, remove_last_relu=False)
        self.layer2 = self._make_layer(
            block_type, 128, layers[1], stride=2, norm_layer=norm_layer, remove_last_relu=False)
        self.layer3 = self._make_layer(
            block_type, 256, layers[2], stride=2, norm_layer=norm_layer, remove_last_relu=False)
        self.layer4 = self._make_layer(
            block_type, 512, layers[3], stride=2, norm_layer=norm_layer, remove_last_relu=remove_last_relu)
        self.remove_last_relu = remove_last_relu
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # this variable is added for compatibility reason
        self.pool = self.avgpool
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block_type, planes, blocks, stride=1, norm_layer=None, remove_last_relu=False):
        if block_type == 'basic':
            block = BasicBlock
        elif block_type == 'bottleneck':
            block = Bottleneck

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                CBN(planes * block.expansion, self.attributes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.attributes,
                            self.base_width, norm_layer, remove_last_relu=False))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks-1):
            layers.append(block(self.inplanes, planes, self.attributes, groups=self.groups, 
                                base_width=self.base_width, norm_layer=norm_layer, 
                                remove_last_relu=False))

        layers.append(block(self.inplanes, planes, self.attributes, groups=self.groups,
                            base_width=self.base_width, norm_layer=norm_layer,
                            remove_last_relu=remove_last_relu))

        return Sequential(*layers)

    def feature_maps(self, x, attributes):
        x = self.layer0(x, attributes)
        x = self.layer1(x, attributes)
        x = self.layer2(x, attributes)
        x = self.layer3(x, attributes)
        x = self.layer4(x, attributes)
        return x

    def feature(self, x):
        x = self.feature_maps(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        x = self.feature(x)
        return x


def resnet18(attributes, **kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet('basic', [2, 2, 2, 2], attributes, **kwargs)
    return model


def resnet20(attributes, **kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet('basic', [2, 2, 2, 3], attributes, **kwargs)
    return model


def resnet34(attributes, **kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet('basic', [3, 4, 6, 3], attributes, **kwargs)
    return model


def resnet50(attributes,**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet('bottleneck', [3, 4, 6, 3], attributes, **kwargs)
    return model


def resnet101(attributes, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet('bottleneck', [3, 4, 23, 3], attributes, **kwargs)
    return model


def resnet152(attributes, **kwargs):
    """Constructs a ResNet-152 model.
    """
    model = ResNet('bottleneck', [3, 8, 36, 3], attributes, **kwargs)
    return model