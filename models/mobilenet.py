import torch.nn as nn
import structured.fully_connected as fc

models = {"ImageNet": "mobilenet"}


# Implementation based on https://github.com/marvis/pytorch-mobilenet
class MobileNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=1000, conv2d=fc.Conv2d,
                 args=None):
        super(MobileNet, self).__init__()
        self.width = args.width_multiplier
        self.num_classes = num_classes

        self.model = nn.Sequential(
            self.conv_bn(input_channels, 32*self.width, 2, args),
            self.conv_dw(32*self.width, 64*self.width, 1, conv2d, args),
            self.conv_dw(64*self.width, 128*self.width, 2, conv2d, args),
            self.conv_dw(128*self.width, 128*self.width, 1, conv2d, args),
            self.conv_dw(128*self.width, 256*self.width, 2, conv2d, args),
            self.conv_dw(256*self.width, 256*self.width, 1, conv2d, args),
            self.conv_dw(256*self.width, 512*self.width, 2, conv2d, args),
            self.conv_dw(512*self.width, 512*self.width, 1, conv2d, args),
            self.conv_dw(512*self.width, 512*self.width, 1, conv2d, args),
            self.conv_dw(512*self.width, 512*self.width, 1, conv2d, args),
            self.conv_dw(512*self.width, 512*self.width, 1, conv2d, args),
            self.conv_dw(512*self.width, 512*self.width, 1, conv2d, args),
            self.conv_dw(512*self.width, 1024*self.width, 2, conv2d, args),
            self.conv_dw(1024*self.width, 1024*self.width, 1, conv2d, args),
            nn.AvgPool2d(7),
        )

        self.fc = conv2d(1024*self.width, num_classes, kernel_size=1, args=args)

    @staticmethod
    def conv_bn(inputs, outputs, stride, args):
        # Removed batch-norm layers because they are included within the
        # modified convolution implementations.
        return nn.Sequential(
            fc.Conv2d(inputs, outputs, 3, stride, 1, bias=False, args=args),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def conv_dw(inputs, outputs, stride, conv2d, args):
        # Removed batch-norm layers because they are included within the
        # modified convolution implementations.
        return nn.Sequential(
            conv2d(inputs, inputs, 3, stride, 1, groups=inputs, bias=False,
                   args=args),
            nn.ReLU(inplace=True),

            conv2d(inputs, outputs, 1, 1, 0, bias=False, args=args),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, int(1024 * self.width), 1, 1)
        x = self.fc(x)
        return x.view(-1, self.num_classes)


def mobilenet(**kwargs):
    model = MobileNet(**kwargs)
    return model
