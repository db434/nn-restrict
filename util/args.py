import argparse

import torch
import torch.distributed as dist

import models
import structured

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

conv_types = sorted(structured.conv2d_types.keys())


def parse_args(dataset):
    """An argument parser with some behaviour common to all datasets."""

    parser = argparse.ArgumentParser(description=dataset.name + " training")
    
    # Architecture parameters.
    parser.add_argument('--arch', '-a', metavar='ARCH',
                        default=dataset.default_model, choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names)
                             + ' (default: ' + dataset.default_model + ')')
    parser.add_argument('--width-multiplier', '-w', default=1, type=float,
                        help='ratio of channels compared to base model')
    parser.add_argument('--conv-type', '-c', metavar='CONV', default='fc',
                        type=str, choices=conv_types,
                        help='type of convolution to use: ' + ' | '.join(
                            conv_types))
                        
    # Training options.
    parser.add_argument('--epochs', default=dataset.default_epochs, type=int,
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--use-restarts', action='store_true',
                        help='periodically reset learning rate')
    parser.add_argument('--restart-period', default=0, type=int,
                        help='restart learning rate after this many epochs')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--resume', action='store_true',
                        help='resume from latest checkpoint in save_dir')
    parser.add_argument('--pretrained', action='store_true',
                        help='use pre-trained model')
                        
    # Distributed computing options.
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='gloo', type=str,
                        help='distributed backend')
    
    # Number representations.
    parser.add_argument('--grad-noise', default=0.0, type=float,
                        help='add random noise to all gradients')
    parser.add_argument('--grad-precision', default=0.0, type=float,
                        help='set smallest possible difference between values')
    parser.add_argument('--grad-min', default=0.0, type=float,
                        help='zero all gradients with magnitudes below min')
    parser.add_argument('--grad-max', default=0.0, type=float,
                        help='clip all gradient magnitudes to max')
    parser.add_argument('--act-noise', default=0.0, type=float,
                        help='add random noise to all activations')
    parser.add_argument('--act-precision', default=0.0, type=float,
                        help='set smallest possible difference between values')
    parser.add_argument('--act-min', default=0.0, type=float,
                        help='zero all activations with magnitudes below min')
    parser.add_argument('--act-max', default=0.0, type=float,
                        help='clip all activation magnitudes to max')
    parser.add_argument('--weight-noise', default=0.0, type=float,
                        help='add random noise to all weights')
    parser.add_argument('--weight-precision', default=0.0, type=float,
                        help='set smallest possible difference between values')
    parser.add_argument('--weight-min', default=0.0, type=float,
                        help='zero all weights with magnitudes below min')
    parser.add_argument('--weight-max', default=0.0, type=float,
                        help='clip all weight magnitudes to max')
    
    # Storage.
    parser.add_argument('--save-dir', default='.', type=str,
                        help='directory to save models in')
    parser.add_argument('--dump-dir', default=None, type=str,
                        help='directory to dump all weights, acts and grads')
    parser.add_argument('--undump-dir', default=None, type=str,
                        help='directory from which to restore dumped weights')
                        
    # Miscellaneous.
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--print-freq', '-p', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('-e', '--evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-s', '--stats', action='store_true', default=False,
                        help='print stats about the model and exit')
    parser.add_argument('--gradients', action='store_true', default=False,
                        help='print stats about gradients and exit')
    parser.add_argument('--random-seed', metavar='SEED', default=None, type=int,
                        help='initialise the random number generator')
    
    # Some tidying and handling of simple parameters.
    args = parser.parse_args()
    
    # Use int representation if possible - it prints more nicely.
    if args.width_multiplier == int(args.width_multiplier):
        args.width_multiplier = int(args.width_multiplier)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size)
    
    if args.random_seed:
        torch.manual_seed(args.random_seed)
    
    if args.restart_period > 0:
        args.use_restarts = True
    
    if args.resume:
        assert args.save_dir
    
    assert args.conv_type in structured.conv2d_types
    
    return args
