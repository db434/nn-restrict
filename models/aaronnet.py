import torch.nn as nn
import structured.fully_connected as fc

models = {"CIFAR-10": ["aaronnet", "aaronnet_v2"]}


class AaronNet(nn.Module):

    def __init__(self, input_channels=3, num_classes=10, conv2d=fc.Conv2d,
                 args=None):
        super(AaronNet, self).__init__()
        self.classes = num_classes
        w = args.width_multiplier
        
        self.features = nn.Sequential(
            fc.Conv2d(input_channels, 128*w, kernel_size=3, padding=1,
                      args=args),
            nn.ReLU(inplace=True),
            conv2d(128*w, 128*w, kernel_size=3, padding=1, args=args),
            nn.ReLU(inplace=True),
            conv2d(128*w, 128*w, kernel_size=3, padding=1, args=args),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            conv2d(128*w, 128*w, kernel_size=3, padding=1, args=args),
            nn.ReLU(inplace=True),
            conv2d(128*w, 128*w, kernel_size=3, padding=1, args=args),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            conv2d(128*w, 128*w, kernel_size=3, padding=1, args=args),
            nn.ReLU(inplace=True),
            conv2d(128*w, 128*w, kernel_size=3, padding=1, args=args),
            nn.ReLU(inplace=True),
            
            nn.AvgPool2d(kernel_size=7, stride=2, padding=0),
            
            nn.Dropout(),
            conv2d(128*w, num_classes, kernel_size=1, padding=0, args=args),
        )

    def forward(self, x):
        x = self.features(x)
        return x.view(-1, self.classes)


class AaronNet2(nn.Module):

    def __init__(self, input_channels=3, num_classes=10, conv2d=fc.Conv2d,
                 args=None):
        super(AaronNet2, self).__init__()
        self.classes = num_classes
        w = args.width_multiplier
        
        self.features = nn.Sequential(
            fc.Conv2d(input_channels, 64*w, kernel_size=3, padding=1,
                      args=args),
            nn.ReLU(inplace=True),
            conv2d(64*w, 64*w, kernel_size=3, padding=1, args=args),
            nn.ReLU(inplace=True),
            conv2d(64*w, 128*w, kernel_size=3, padding=1, stride=2, args=args),
            nn.ReLU(inplace=True),            
            conv2d(128*w, 128*w, kernel_size=3, padding=1, args=args),
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=0.5),
            
            conv2d(128*w, 128*w, kernel_size=3, padding=1, args=args),
            nn.ReLU(inplace=True),            
            conv2d(128*w, 192*w, kernel_size=3, padding=1, stride=2, args=args),
            nn.ReLU(inplace=True),
            conv2d(192*w, 192*w, kernel_size=3, padding=1, args=args),
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=0.5),
            
            conv2d(192*w, 192*w, kernel_size=3, padding=1, args=args),
            nn.ReLU(inplace=True),
            
            nn.AvgPool2d(kernel_size=8, padding=0),
            
            conv2d(192*w, num_classes, kernel_size=1, padding=0, args=args),
        )

    def forward(self, x):
        x = self.features(x)
        batch, channels, width, height = x.size()
        return x.view(batch, self.classes)


def aaronnet(**kwargs):
    r"""CIFAR-10 model from Aaron as described in test case 1 here:
    https://github.com/admk/mayo-dev/issues/42
    https://github.com/admk/mayo-dev/blob/develop/models/cifarnet_had.yaml
    
    There's potential for also adding the additional convolution layers that he
    uses in subsequent test cases.
    
    Default version gets >91% accuracy, even with severe overfitting.
    """
    model = AaronNet(**kwargs)
    return model


def aaronnet_v2(**kwargs):
    r"""Optimised CIFAR-10 model from Aaron. Fewer parameters, same accuracy.
    
    Default version gets ~93% accuracy.
    """
    model = AaronNet2(**kwargs)
    return model
