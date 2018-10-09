import torch
import torch.nn as nn
import torch.nn.init as init

import structured.fully_connected as fc


models = {"ImageNet": ["squeezenet1_0", "squeezenet1_1"]}


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes, conv2d, args):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = conv2d(inplanes, squeeze_planes, kernel_size=1,
                              args=args)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = conv2d(squeeze_planes, expand1x1_planes,
                                kernel_size=1, args=args)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = conv2d(squeeze_planes, expand3x3_planes,
                                kernel_size=3, padding=1, args=args)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Module):

    def __init__(self, version=1.0, input_channels=3, num_classes=1000,
                 conv2d=fc.Conv2d, args=None):
        super(SqueezeNet, self).__init__()
        w = args.width_multiplier  # Very short name because used often below
        
        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=version))
        self.num_classes = num_classes
        if version == 1.0:
            self.features = nn.Sequential(
                fc.Conv2d(input_channels, 96*w, kernel_size=7, stride=2,
                          args=args),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96*w,  16*w, 64*w,  64*w,  conv2d, args),
                Fire(128*w, 16*w, 64*w,  64*w,  conv2d, args),
                Fire(128*w, 32*w, 128*w, 128*w, conv2d, args),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256*w, 32*w, 128*w, 128*w, conv2d, args),
                Fire(256*w, 48*w, 192*w, 192*w, conv2d, args),
                Fire(384*w, 48*w, 192*w, 192*w, conv2d, args),
                Fire(384*w, 64*w, 256*w, 256*w, conv2d, args),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512*w, 64*w, 256*w, 256*w, conv2d, args),
            )
        else:
            self.features = nn.Sequential(
                fc.Conv2d(input_channels, 64*w, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64*w,  16*w, 64*w,  64*w,  conv2d, args),
                Fire(128*w, 16*w, 64*w,  64*w,  conv2d, args),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128*w, 32*w, 128*w, 128*w, conv2d, args),
                Fire(256*w, 32*w, 128*w, 128*w, conv2d, args),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256*w, 48*w, 192*w, 192*w, conv2d, args),
                Fire(384*w, 48*w, 192*w, 192*w, conv2d, args),
                Fire(384*w, 64*w, 256*w, 256*w, conv2d, args),
                Fire(512*w, 64*w, 256*w, 256*w, conv2d, args),
            )
        # Final convolution is initialized differently form the rest
        final_conv = conv2d(512*w, self.num_classes, kernel_size=1, args=args)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),  # TODO remove this dropout?
            final_conv,
            nn.ReLU(inplace=True),
            nn.AvgPool2d(13)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal(m.weight.data, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)


def squeezenet1_0(**kwargs):
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.
    """
    model = SqueezeNet(version=1.0, **kwargs)
    return model


def squeezenet1_1(**kwargs):
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.
    """
    model = SqueezeNet(version=1.1, **kwargs)
    return model
