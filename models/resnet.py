import torch.nn as nn
import math

import structured.fully_connected as fc

models = {"CIFAR-10": ["resnet20", "resnet32", "resnet44", "resnet56",
                       "resnet110"],
          "ImageNet": ["resnet18", "resnet34", "resnet50", "resnet101",
                       "resnet152"]}


def conv3x3(in_planes, out_planes, conv2d, args, stride=1):
    """3x3 convolution with padding"""
    return conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                  padding=1, bias=False, args=args)


class BasicBlock(nn.Module):
    # db434: removed batch-norm layers because this is included within the
    # modified convolution implementations.
    
    expansion = 1

    def __init__(self, inplanes, planes, conv2d, args, stride=1,
                 downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, conv2d, args, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, conv2d, args)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Default Bottleneck module using fully-connected convolution."""
    # db434: removed batch-norm layers because this is included within the
    # modified convolution implementations.
    
    expansion = 4

    def __init__(self, inplanes, planes, conv2d, args, stride=1,
                 downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv2d(inplanes, planes, kernel_size=1, bias=False,
                            args=args)
        self.conv2 = conv2d(planes, planes, kernel_size=3, stride=stride,
                            padding=1, bias=False, args=args)
        self.conv3 = conv2d(planes, planes * 4, kernel_size=1, bias=False,
                            args=args)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, input_channels=3, num_classes=1000,
                 conv2d=fc.Conv2d, args=None):
        # db434: removed batch-norm layers because this is included within the
        # modified convolution implementations.
        w = args.width_multiplier
        self.inplanes = 64 * w
        
        super(ResNet, self).__init__()
        self.conv1 = fc.Conv2d(input_channels, 64*w, kernel_size=7, stride=2,
                               padding=3, bias=False, args=args)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64*w,  layers[0], conv2d, args)
        self.layer2 = self._make_layer(block, 128*w, layers[1], conv2d, args,
                                       stride=2)
        self.layer3 = self._make_layer(block, 256*w, layers[2], conv2d, args,
                                       stride=2)
        self.layer4 = self._make_layer(block, 512*w, layers[3], conv2d, args,
                                       stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = conv2d(512 * w * block.expansion, num_classes,
                         kernel_size=1, args=args)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, conv2d, args, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv2d(self.inplanes, planes * block.expansion,
                       kernel_size=1, stride=stride, bias=False, args=args)
            )

        layers = [block(self.inplanes, planes, conv2d, args, stride,
                        downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv2d, args))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1, 1, 1)
        x = self.fc(x)

        return x.view(x.size(0), -1)


class ResNetCifar(nn.Module):
    """Specialisation of ResNet for the CIFAR-10 dataset. Can still be applied
    to ImageNet, but won't downsample very much.
    
    The first layer is different, and there are only three blocks of layers
    instead of four.
    """

    def __init__(self, block, layers, input_channels=3, num_classes=10,
                 conv2d=fc.Conv2d, args=None):
        
        w = args.width_multiplier
        self.inplanes = 16 * w
        
        super(ResNetCifar, self).__init__()
        self.conv1 = fc.Conv2d(input_channels, 16*w, kernel_size=3,
                               padding=1, bias=False, args=args)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16*w, layers[0], conv2d, args)
        self.layer2 = self._make_layer(block, 32*w, layers[1], conv2d, args,
                                       stride=2)
        self.layer3 = self._make_layer(block, 64*w, layers[2], conv2d, args,
                                       stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = conv2d(64*w*block.expansion, num_classes,
                         kernel_size=1, args=args)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, conv2d, args, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = conv2d(self.inplanes, planes * block.expansion,
                                kernel_size=1, stride=stride, bias=False,
                                args=args)

        layers = [block(self.inplanes, planes, conv2d, args, stride,
                        downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv2d, args))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1, 1, 1)
        x = self.fc(x)

        return x.view(x.size(0), -1)


def resnet18(**kwargs):
    """Constructs a ResNet-18 model."""
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet20(**kwargs):
    """Constructs a ResNet-20 model for CIFAR-10. Paper claims 8.75% error."""
    model = ResNetCifar(BasicBlock, [3, 3, 3], **kwargs)
    return model


def resnet32(**kwargs):
    """Constructs a ResNet-32 model for CIFAR-10. Paper claims 7.51% error."""
    model = ResNetCifar(BasicBlock, [5, 5, 5], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model."""
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet44(**kwargs):
    """Constructs a ResNet-44 model for CIFAR-10. Paper claims 7.17% error."""
    model = ResNetCifar(BasicBlock, [7, 7, 7], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    
Pretrained: 23.85%/7.13% top1/top5 error."""
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet56(**kwargs):
    """Constructs a ResNet-56 model for CIFAR-10. Paper claims 6.97% error."""
    model = ResNetCifar(BasicBlock, [9, 9, 9], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model."""
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet110(**kwargs):
    """Constructs a ResNet-110 model for CIFAR-10. Paper claims 6.43% error."""
    model = ResNetCifar(BasicBlock, [18, 18, 18], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-152 model."""
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model
