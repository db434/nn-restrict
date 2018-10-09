import torch.nn as nn
import structured.fully_connected as fc

models = {"MNIST": ["lenet5", "mnistnet"]}


class LeNet5(nn.Module):
    """
    This doesn't quite match the original, but isn't far off:
      * Batch norm layers are included in convolution layers.
      * The second convolution layer should include some sparse connections.

    http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
    """
    def __init__(self, input_channels=1, num_classes=10, conv2d=fc.Conv2d,
                 args=None):
        super(LeNet5, self).__init__()
        self.width = args.width_multiplier
        w = self.width  # Super short name
        self.num_classes = num_classes

        self.model = nn.Sequential(
            fc.Conv2d(input_channels, 6*w, kernel_size=5, padding=0, args=args),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Tanh(),
            conv2d(6*w, 16*w, kernel_size=5, padding=0, args=args),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Tanh(),
            conv2d(16*w, 120*w, kernel_size=5, padding=0, args=args),
            nn.Tanh(),
            conv2d(120*w, 84*w, kernel_size=1, args=args),
            nn.Tanh(),
            conv2d(84*w, num_classes, kernel_size=1, args=args)
        )

    def forward(self, x):
        x = self.model(x)
        return x.view(x.size(0), self.num_classes)


class MnistNet(nn.Module):
    """
    This is based on LeNet5, but is much larger.
    """

    def __init__(self, input_channels=1, num_classes=10, conv2d=fc.Conv2d,
                 args=None):
        super(MnistNet, self).__init__()
        self.width = args.width_multiplier
        w = self.width  # Super short name
        self.num_classes = num_classes

        self.features = nn.Sequential(
            fc.Conv2d(input_channels, 32*w, kernel_size=5, padding=0,
                      args=args),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            conv2d(32*w, 64*w, kernel_size=5, padding=2, args=args),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            conv2d(64*7*7*w, 1024*w, kernel_size=1, args=args),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            conv2d(1024*w, num_classes, kernel_size=1, args=args)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 64 * 7 * 7 * self.width, 1, 1)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)


def lenet5(**kwargs):
    """~99% accuracy."""
    model = LeNet5(**kwargs)
    return model


def mnistnet(**kwargs):
    """~99.3% accuracy"""
    model = MnistNet(**kwargs)
    return model
