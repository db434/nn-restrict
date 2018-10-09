from functools import reduce
import torch

from modifiers.modules import Quantisable
from structured import convert_to_conv
from util import log

from . import lenet, aaronnet, alexnet, densenet, mobilenet, resnet, \
    squeezenet, vgg, wlm

submodules = [lenet, aaronnet, alexnet, densenet, mobilenet, resnet,
              squeezenet, vgg, wlm]


def get_model_names(dataset=None):
    names = []
    for module in submodules:
        models = module.models

        if dataset is None:
            # Get all available model names
            for data in models.values():
                if isinstance(data, str):
                    names.append(data)
                else:
                    assert isinstance(data, list)
                    names += data
        elif dataset.name in models:
            # Just get names of models made for the given dataset.
            data = models[dataset.name]
            if isinstance(data, str):
                names.append(data)
            else:
                assert isinstance(data, list)
                names += data

    return names


def get_model(name, distributed=False, use_cuda=True, **kwargs):
    log.info("Creating model '{}'".format(name))

    model = None
    for module in submodules:
        if name in dir(module):
            model = getattr(module, name)(**kwargs)
            break
    assert model is not None

    log.info("Model has", count_parameters(model), "parameters")

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

    model = Quantisable(model)

    # Hack. RNNs have their own particular initialisation, so don't
    # reinitialise if any RNN layers are found.
    rnn = False
    for m in model.modules():
        if isinstance(m, convert_to_conv.RNNBase):
            rnn = True
            break

    # Initialise the convolution weights according to their fan-in.
    # Butterfly convolution consistently performs a little worse without
    # this, so apply it to all models to be fair.
    if not rnn:
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight.data)

    return model


def count_parameters(model):
    count = 0
    for param in model.parameters():
        size = param.size()
        count += reduce(lambda x, y: x*y, size)
    return count
