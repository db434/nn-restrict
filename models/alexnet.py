import torch.nn as nn
import structured.fully_connected as fc

models = {"ImageNet": "alexnet"}


class AlexNet(nn.Module):

    def __init__(self, input_channels=3, num_classes=1000, conv2d=fc.Conv2d,
                 args=None):
        super(AlexNet, self).__init__()
        self.width = args.width_multiplier
        w = self.width  # Super short name
        self.num_classes = num_classes

        self.features = nn.Sequential(
            fc.Conv2d(input_channels, 64*w, kernel_size=11, stride=4,
                      padding=2, args=args),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            conv2d(64*w, 192*w, kernel_size=5, padding=2, args=args),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            conv2d(192*w, 384*w, kernel_size=3, padding=1, args=args),
            nn.ReLU(inplace=True),
            conv2d(384*w, 256*w, kernel_size=3, padding=1, args=args),
            nn.ReLU(inplace=True),
            conv2d(256*w, 256*w, kernel_size=3, padding=1, args=args),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            # nn.Linear(256 * 6 * 6 * w, 4096),
            conv2d(int(256*w)*6*6, 4096*w, kernel_size=1, args=args),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            # nn.Linear(4096, 4096),
            conv2d(4096*w, 4096*w, kernel_size=1, args=args),
            nn.ReLU(inplace=True),
            # nn.Linear(4096, num_classes),
            conv2d(4096*w, num_classes, kernel_size=1, args=args),
            # fc.Conv2d(4096 * w, num_classes, kernel_size=1),
        )

    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0), 256 * 6 * 6 * self.width)
        x = x.view(x.size(0), int(256 * self.width) * 6 * 6, 1, 1)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)


def alexnet(**kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    
    Pretrained: 43.45%/20.91% top1/top5 error.
    """
    model = AlexNet(**kwargs)
    return model
