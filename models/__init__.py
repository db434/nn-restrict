__all__ = ["mnist", "alexnet", "densenet", "resnet", "squeezenet", "vgg",
           "aaronnet"]

# MNIST
# from .mnist import *

# CIFAR-10
from .aaronnet import *

# ImageNet
from .alexnet import *
from .densenet import *
from .resnet import *
from .squeezenet import *
from .vgg import *

from functools import reduce
import torch


def get_model(name, distributed=False, use_cuda=True, **kwargs):
    print("=> creating model '{}'".format(name))
    model = globals()[name](**kwargs)
    print("Model has", count_parameters(model), "parameters")

    # Automatically make models run on multiple GPUs and/or multiple machines.
    # I assume that use_cuda=False also means "don't use GPUs".
    if not distributed:
        if use_cuda:
            # Some magic from the original ImageNet script. Not sure what the
            # purpose is.
            if name.startswith('alexnet') or name.startswith('vgg'):
                model.features = torch.nn.DataParallel(model.features)
                model = model.cuda()
            else:
                model = torch.nn.DataParallel(model)
                model = model.cuda()
    else:
        if use_cuda:
            model = model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    
    return model


def count_parameters(model):
    count = 0
    for param in model.parameters():
        size = param.size()
        count += reduce(lambda x, y: x*y, size)
    return count
