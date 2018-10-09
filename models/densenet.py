import torch
import torch.nn as nn
import torch.nn.functional
from collections import OrderedDict

import structured.fully_connected as fc

__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201',
           'densenet161']


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate,
                 conv2d):
        # db434: removed batch-norm layers because this is included within the
        # modified convolution implementations.
        super(_DenseLayer, self).__init__()
        # self.add_module('norm.1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu.1', nn.ReLU(inplace=True)),
        self.add_module('conv.1', conv2d(num_input_features, bn_size *
                                         growth_rate, kernel_size=1, stride=1,
                                         bias=False)),
        # self.add_module('norm.2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu.2', nn.ReLU(inplace=True)),
        self.add_module('conv.2', conv2d(bn_size * growth_rate, growth_rate,
                                         kernel_size=3, stride=1, padding=1,
                                         bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = nn.functional.dropout(new_features, p=self.drop_rate,
                                                 training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
                 drop_rate, conv2d):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate, bn_size, drop_rate, conv2d)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, conv2d):
        # db434: removed batch-norm layer because this is included within the
        # modified convolution implementations.
        super(_Transition, self).__init__()
        # self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', conv2d(num_input_features, num_output_features,
                                       kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks"
    <https://arxiv.org/pdf/1608.06993.pdf>`

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first
        convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000,
                 input_channels=3, width_multiplier=1, conv2d=fc.Conv2d):

        super(DenseNet, self).__init__()
        self.width = width_multiplier

        num_init_features *= self.width
        growth_rate *= self.width

        # First convolution
        # db434: removed batch-norm layer because this is included within the
        # modified convolution implementations.
        self.features = nn.Sequential(OrderedDict([
            ('conv0', fc.Conv2d(input_channels, num_init_features,
                                kernel_size=7, stride=2, padding=3,
                                bias=False)),
            # ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate,
                                drop_rate=drop_rate, conv2d=conv2d)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2,
                                    conv2d=conv2d)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        # self.classifier = nn.Linear(num_features, num_classes)
        self.classifier = conv2d(num_features, num_classes, kernel_size=1)

    def forward(self, x):
        features = self.features(x)
        out = nn.functional.relu(features, inplace=True)
        # out = F.avg_pool2d(out, kernel_size=7).view(features.size(0), -1)
        out = nn.functional.avg_pool2d(out, kernel_size=7).view(
            features.size(0), -1, 1, 1)
        out = self.classifier(out)
        return out.view(out.size(0), -1)


def densenet121(**kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks"
    <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    model = DenseNet(num_init_features=64, growth_rate=32,
                     block_config=(6, 12, 24, 16),
                     **kwargs)
    return model


def densenet169(**kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks"
    <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    model = DenseNet(num_init_features=64, growth_rate=32,
                     block_config=(6, 12, 32, 32),
                     **kwargs)
    return model


def densenet201(**kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks"
    <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    model = DenseNet(num_init_features=64, growth_rate=32,
                     block_config=(6, 12, 48, 32),
                     **kwargs)
    return model


def densenet161(**kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks"
    <https://arxiv.org/pdf/1608.06993.pdf>`
    
    Pretrained: 22.35%/6.20% top1/top5 error.
    """
    model = DenseNet(num_init_features=96, growth_rate=48,
                     block_config=(6, 12, 36, 24),
                     **kwargs)
    return model
