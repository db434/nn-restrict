import argparse

import torch
import torch.distributed as dist

import models
import structured


def parse_args(dataset):
    """An argument parser with some behaviour common to all datasets."""

    model_names = sorted(models.get_model_names(dataset=dataset))
    conv_types = sorted(structured.conv2d_types.keys())

    parser = argparse.ArgumentParser(description=dataset.name + " training")
    
    # Architecture parameters.
    arch = parser.add_argument_group(title="Architecture parameters")
    arch.add_argument('--arch', '-a', metavar='ARCH',
                      default=dataset.default_model, choices=model_names,
                      help='model architecture: ' + ' | '.join(model_names)
                           + ' (default: ' + dataset.default_model + ')')
    arch.add_argument('--width-multiplier', '-w', default=1, type=float,
                      metavar="W",
                      help='ratio of channels compared to base model')
    arch.add_argument('--conv-type', '-c', metavar='CONV', default='fc',
                      type=str, choices=conv_types,
                      help='type of convolution to use: ' + ' | '.join(
                          conv_types))
    arch.add_argument('--min-bfly-size', metavar='SIZE', default=2, type=int,
                      help='minimum allowed butterfly size')

    # Storage.
    store = parser.add_argument_group(title="Storage options")
    store.add_argument('--save-dir', default='.', type=str,
                       help='directory to save models in')
    store.add_argument('--dump-dir', default=None, type=str,
                       help='directory to dump all weights, acts and grads')
    store.add_argument('--undump-dir', default=None, type=str,
                       help='directory from which to restore dumped weights')
                        
    # Training options.
    train = parser.add_argument_group(title="Training options")
    train.add_argument('--epochs', default=dataset.default_epochs, type=int,
                       metavar='N', help='number of total epochs to run')
    train.add_argument('--start-epoch', default=0, type=int, metavar='N',
                       help='manual epoch number (useful on restarts)')
    train.add_argument('--use-restarts', action='store_true',
                       help='periodically reset learning rate')
    train.add_argument('--restart-period', metavar='EPOCHS', type=int,
                       help='restart learning rate after this many epochs')
    train.add_argument('-b', '--batch-size', default=256, type=int,
                       metavar='N', help='mini-batch size (default: 256)')
    train.add_argument('--lr', '--learning-rate',
                       default=dataset.default_lr, type=float,
                       metavar='LR', help='initial learning rate')
    train.add_argument('--momentum', default=0.9, type=float, metavar='M',
                       help='momentum')
    train.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                       metavar='W', help='weight decay (default: 1e-4)')
    train.add_argument('--distill', action='store_true',
                       help='train using knowledge distillation')
    train.add_argument('--resume', action='store_true',
                       help='resume from latest checkpoint in save_dir')
    train.add_argument('--model-file', metavar='PATH', type=str,
                       help='load model checkpoint from the named file')
    train.add_argument('--pretrained', action='store_true',
                       help='use pre-trained model')
    
    # Number representations.
    number = parser.add_argument_group(title="Number representations")
    number.add_argument('--grad-noise', default=0.0, type=float,
                        help='add random noise to all gradients')
    number.add_argument('--grad-precision', default=0.0, type=float,
                        help='set smallest possible difference between values')
    number.add_argument('--grad-min', default=0.0, type=float,
                        help='zero all gradients with magnitudes below min')
    number.add_argument('--grad-max', default=0.0, type=float,
                        help='clip all gradient magnitudes to max')
    number.add_argument('--act-noise', default=0.0, type=float,
                        help='add random noise to all activations')
    number.add_argument('--act-precision', default=0.0, type=float,
                        help='set smallest possible difference between values')
    number.add_argument('--act-min', default=0.0, type=float,
                        help='zero all activations with magnitudes below min')
    number.add_argument('--act-max', default=0.0, type=float,
                        help='clip all activation magnitudes to max')
    number.add_argument('--weight-noise', default=0.0, type=float,
                        help='add random noise to all weights')
    number.add_argument('--weight-precision', default=0.0, type=float,
                        help='set smallest possible difference between values')
    number.add_argument('--weight-min', default=0.0, type=float,
                        help='zero all weights with magnitudes below min')
    number.add_argument('--weight-max', default=0.0, type=float,
                        help='clip all weight magnitudes to max')

    # Data collection.
    collect = parser.add_argument_group(title="Data collection")
    collect.add_argument('-s', '--stats', action='store_true', default=False,
                         help='print stats about the model and exit')
    collect.add_argument('--gradients', action='store_true', default=False,
                         help='print stats about gradients and exit')
    collect.add_argument('--dump-weights', action='store_true', default=False,
                         help='dump all weights to `dump-dir`')
    collect.add_argument('--dump-acts', action='store_true', default=False,
                         help='dump all activations to `dump-dir`')
    collect.add_argument('--dump-grads', action='store_true', default=False,
                         help='dump all gradients to `dump-dir`')

    # Distributed computing options.
    distribute = parser.add_argument_group(title="Distributed computing")
    distribute.add_argument('--world-size', default=1, type=int,
                            help='number of distributed processes')
    distribute.add_argument('--dist-url', default='tcp://224.66.41.62:23456',
                            type=str,
                            help='url used to set up distributed training')
    distribute.add_argument('--dist-backend', default='gloo', type=str,
                            help='distributed backend')

    # Text generation.
    generate = parser.add_argument_group(title="Text generation")
    generate.add_argument('--generate', action='store_true', default=False,
                          help='generate text')
    generate.add_argument('--words', default=100, type=int,
                          help='number of words to generate')
    generate.add_argument('--temperature', default=1.0, type=float,
                          help='higher temperature gives higher diversity')
                        
    # Miscellaneous.
    misc = parser.add_argument_group(title="Miscellaneous")
    misc.add_argument('--no-cuda', action='store_true', default=False,
                      help='disables CUDA training')
    misc.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                      help='number of data loading workers (default: 16)')
    misc.add_argument('--print-freq', '-p', default=100, type=int, metavar='N',
                      help='print progress every N batches (default: 100)')
    misc.add_argument('-e', '--evaluate', action='store_true',
                      help='evaluate model on validation set')
    misc.add_argument('--random-seed', metavar='SEED', default=None, type=int,
                      help='initialise the random number generator')
    misc.add_argument('--list-arches', action='store_true', default=False,
                      help='list all model architectures for all datasets')
    
    # Some tidying and handling of simple parameters.
    args = parser.parse_args()

    if args.list_arches:
        print("Any architecture can be applied to any dataset, but spatial "
              "downsampling (e.g. pooling) may be inappropriate if image "
              "sizes differ.")
        print(", ".join(sorted(models.get_model_names())))
        exit(0)
    
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
    
    if args.restart_period is not None:
        args.use_restarts = True

    if args.resume:
        assert args.save_dir or args.model_file
    
    assert args.conv_type in structured.conv2d_types

    # A bit of a hack to add more dummy arguments for text datasets.
    if hasattr(dataset, "num_tokens"):
        args.num_tokens = dataset.num_tokens()
    
    return args
