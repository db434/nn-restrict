import csv
import numpy
import os
import shutil
import time
import torch

from modifiers.modules import Quantisable
from util import log

"""Data written to a log file at the end of each epoch."""
log_data = ["Time", "Epoch", "Train loss", "Train top1",
            "Train top5", "Val loss", "Val top1", "Val top5"]


def load(directory, model_name, model, optimizer):
    """Load a checkpoint from a file.
    
    The given model and optimizer must have identical "shapes" to those saved in
    the checkpoint.
    
    Returns (start epoch, best top-1 precision) and modifies the given model and
    optimizer.
    """
    
    path = os.path.join(directory, model_name + "_check.pth.tar")
    return load_path(path, model, optimizer)


def load_path(path, model, optimizer):
    """Load a checkpoint from a named file."""
    
    if os.path.isfile(path):
        log.info("Loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        log.info("Loaded checkpoint '{}' (epoch {})"
                 .format(path, checkpoint['epoch']))
              
        return start_epoch, best_prec1
    else:
        log.error("No checkpoint found at '{}'".format(path))
        exit(1)


def save(directory, model, model_name, optimizer, epoch, best_prec1, is_best):
    """Save a checkpoint to a file.
    
    directory:  directory to save in
    model:      trained model
    model_name: unique identifier for this model
    optimizer:  optimizer used
    epoch:      number of epochs trained so far
    best_prec1: best top1 accuracy achieved by this network so far
    is_best:    does the current model achieve best_prec1 accuracy?
    """
    # If the model computes using quantised parameters, restore the full
    # precision ones before storing.
    if isinstance(model, Quantisable):
        model.restore_parameters()

    state = {
        'epoch': epoch + 1,
        'arch': model_name,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer':  optimizer.state_dict(),
    }
    
    checkpoint = os.path.join(directory, model_name + "_check.pth.tar")
    torch.save(state, checkpoint)
    if is_best:
        best = os.path.join(directory, model_name + "_best.pth.tar")
        shutil.copyfile(checkpoint, best)


def load_tensor(directory, description, tensor):
    """Load a numpy.ndarray from a file into a given tensor."""
    path = os.path.join(directory, description + ".npy")
    loaded = numpy.load(path)
    loaded = torch.from_numpy(loaded)

    assert tensor.size() == loaded.size()
    tensor.copy_(loaded)


def save_tensor(directory, description, tensor):
    """Save a tensor to a file. Data can be reloaded using numpy.load(file)."""
    path = os.path.join(directory, description + ".npy")
    
    if isinstance(tensor, torch.autograd.Variable):
        tensor = tensor.data
    
    tensor = tensor.cpu().numpy()
    tensor.dump(path)


def log_stats(directory, model_name, epoch, train_loss, train_top1, train_top5,
              val_loss, val_top1, val_top5):
    """Log the current performance to a file in the csv format."""
    
    log_file_path = os.path.join(directory, model_name + ".csv")
    
    # Open new file and insert table header.
    if epoch == 0:
        log_file = open(log_file_path, mode="w", newline="")
        writer = csv.writer(log_file)
        writer.writerow(log_data)
    # Append to existing log.
    else:
        log_file = open(log_file_path, mode="a", newline="")
        writer = csv.writer(log_file)
    
    data = [time.time(), epoch, train_loss, train_top1, train_top5, val_loss,
            val_top1, val_top5]
    writer.writerow(data)
    
    log_file.close()
