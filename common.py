import math
import os
import time
import sys

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim

import models
import modifiers
import structured
import util


def process(dataset):
    """Wrap the main task with some exception handlers. Otherwise a huge,
    unnecessary, parallel stacktrace is printed."""
    try:
        _process(dataset)
    except KeyboardInterrupt:
        # Note that due to a Python multiprocessing issue, the stack trace isn't
        # actually prevented here. https://discuss.pytorch.org/t/9740
        print("Process terminated by user.")
        sys.exit()


def _process(dataset):
    best_top1 = 0
    
    args = util.args.parse_args(dataset)

    conv_type = structured.conv2d_types[args.conv_type]
    unique_name = "{0}-{1}x-{2}".format(args.arch, args.width_multiplier,
                                        args.conv_type)

    # Create model
    model = models.get_model(args.arch, distributed=args.distributed,
                             use_cuda=args.cuda,
                             input_channels=dataset.input_channels(),
                             num_classes=dataset.num_classes(),
                             width_multiplier=args.width_multiplier,
                             conv2d=conv_type)

    # TODO: give each model a dictionary of properties to store this information
    # instead of extending the filename.
    if args.grad_noise > 0:
        unique_name += "_gn" + str(args.grad_noise)
    if args.grad_precision > 0:
        unique_name += "_gp" + str(args.grad_precision)
    if args.grad_min > 0:
        unique_name += "_gt" + str(args.grad_min)
    if args.grad_max > 0:
        unique_name += "_gu" + str(args.grad_max)
    if args.grad_noise > 0 or args.grad_precision > 0 or args.grad_min > 0 or \
       args.grad_max > 0:
        modifiers.numbers.restrict_gradients(model,
                                             noise=args.grad_noise,
                                             precision=args.grad_precision,
                                             minimum=args.grad_min,
                                             maximum=args.grad_max)

    if args.act_noise > 0:
        unique_name += "_an" + str(args.act_noise)
    if args.act_precision > 0:
        unique_name += "_ap" + str(args.act_precision)
    if args.act_min > 0:
        unique_name += "_at" + str(args.act_min)
    if args.act_max > 0:
        unique_name += "_au" + str(args.act_max)
    if args.act_noise > 0 or args.act_precision > 0 or args.act_min > 0 or \
       args.act_max > 0:
        modifiers.numbers.restrict_activations(model,
                                               noise=args.act_noise,
                                               precision=args.act_precision,
                                               minimum=args.act_min,
                                               maximum=args.act_max)

    if args.weight_noise > 0:
        unique_name += "_wn" + str(args.weight_noise)
    if args.weight_precision > 0:
        unique_name += "_wp" + str(args.weight_precision)
    if args.weight_min > 0:
        unique_name += "_wt" + str(args.weight_min)
    if args.weight_max > 0:
        unique_name += "_wu" + str(args.weight_max)
    if args.weight_noise > 0 or args.weight_precision > 0 or \
       args.weight_min > 0 or args.weight_max > 0:
        modifiers.numbers.restrict_weights(model,
                                           noise=args.weight_noise,
                                           precision=args.weight_precision,
                                           minimum=args.weight_min,
                                           maximum=args.weight_max)
    
    # Rescale butterfly weights to avoid "vanishing activations" (and gradients)
    if "butterfly" in args.conv_type:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_uniform(m.weight.data)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(args, model)
    
    if args.cuda:
        criterion = criterion.cuda()

    # Resume from a checkpoint
    if args.undump_dir:
        # Load from numpy arrays. This does not load any optimiser state,
        # so does not support training.
        util.stats.data_restore(args.undump_dir, model)
    elif args.resume or args.evaluate:
        # Proper load from a checkpoint
        start_epoch, best_top1 = \
            util.checkpoint.load(args.save_dir, unique_name, model, optimizer)
        args.start_epoch = start_epoch
    else:
        util.checkpoint.save(args.save_dir, model, unique_name, optimizer,
                             -1, 0, True)

    cudnn.benchmark = True

    # Data loading code
    train_loader, val_loader = dataset.data_loaders(
        args.workers, args.batch_size, distributed=args.distributed)

    # Quick analysis before training.
    if args.evaluate:
        validate(val_loader, model, criterion, args)
    
    if args.stats:
        analyse(train_loader, model, criterion, optimizer, args)
        # print(unique_name, *(util.stats.computation_costs()))
    
    if args.gradients:
        collect_gradients(val_loader, unique_name, model, criterion, optimizer,
                          args.save_dir, args.start_epoch, args)

    if args.evaluate or args.stats or args.gradients:
        return
        
    # val_loss, val_top1, val_top5 = validate(val_loader, model, criterion)
    # util.checkpoint.save(args.save_dir, model, unique_name, optimizer,
    #                      -1, val_top1, True)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            # train_sampler.set_epoch(epoch)
        
        adjust_learning_rate(optimizer, epoch, dataset.default_epoch_period,
                             args)

        # Train for one epoch
        train_loss, train_top1, train_top5 = \
            train(train_loader, model, criterion, optimizer, epoch, args)

        # Evaluate on validation set
        val_loss, val_top1, val_top5 = validate(val_loader, model, criterion,
                                                args)
        prec1 = val_top1

        # Remember best prec@1 and save checkpoint
        is_best = prec1 > best_top1
        best_top1 = max(prec1, best_top1)
        util.checkpoint.save(args.save_dir, model, unique_name, optimizer,
                             epoch, best_top1, is_best)
        
        # Log stats.
        util.checkpoint.log(args.save_dir, unique_name, epoch,
                            train_loss, train_top1, train_top5,
                            val_loss, val_top1, val_top5)


def create_optimizer(args, model):
    return torch.optim.SGD(model.parameters(), args.lr,
                           momentum=args.momentum,
                           weight_decay=args.weight_decay)


def analyse(image_loader, model, criterion, optimizer, args):
    """Train for one batch. Print details about all weights, activations and
    gradients."""
    
    # Register hooks on all Modules so they print their details.
    # util.stats.data_distribution_hooks(model, weights=False,
    # activations=False)
    # util.stats.computation_cost_hooks(model) # This doesn't need training data
    util.stats.data_dump_hooks(model, args.dump_dir)

    # Switch to train mode
    model.train()

    for data, target in image_loader:
        if args.cuda:
            target = target.cuda(async=True)
        
        # Compute output
        output, loss = minibatch(model, data, target, criterion)

        # Update model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        break


def train(image_loader, model, criterion, optimizer, epoch, args):
    """Train for one epoch. Return the current loss, top-1 and top5 accuracies.
    """
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # Switch to train mode
    model.train()

    end = time.time()
    for i, (data, target) in enumerate(image_loader):
        data_time.update(time.time() - end)
    
        if args.cuda:
            target = target.cuda(async=True)

        # Compute output        
        output, loss = minibatch(model, data, target, criterion)

        # Measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], data.size(0))
        top1.update(prec1[0], data.size(0))
        top5.update(prec5[0], data.size(0))

        # Update model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(image_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
    
    return losses.avg, top1.avg, top5.avg


def validate(image_loader, model, criterion, args):
    """Run the validation dataset through the model. Return the current loss,
    top-1 and top5 accuracies."""
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # Switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (data, target) in enumerate(image_loader):
        if args.cuda:
            target = target.cuda(async=True)
            
        # Compute output
        output, loss = minibatch(model, data, target, criterion)

        # Measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], data.size(0))
        top1.update(prec1[0], data.size(0))
        top5.update(prec5[0], data.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(image_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def minibatch(model, input_data, target, criterion):
    """Push one minibatch of data through the model. Return the output of the
    model and the loss."""    
    input_var = torch.autograd.Variable(input_data)
    target_var = torch.autograd.Variable(target)

    # Compute output
    output = model(input_var)
    loss = criterion(output, target_var)
    
#        with torch.cuda.profiler.profile():
#            model(input_var) # Warm up CUDA memory allocator and profiler
#            with torch.autograd.profiler.emit_nvtx():
#                output = model(input_var)
#        exit(1)
    
    return output, loss


def collect_gradients(image_loader, model_name, model, criterion, optimiser,
                      directory, epoch, args):
                      
    basename = os.path.join(directory, model_name + "_epoch" + str(epoch))
    checkpoint = basename + ".pth.tar"
    log = basename + ".gradients"
    
    # Load this epoch's checkpoint.
    assert os.path.isfile(checkpoint)    
    util.checkpoint.load_path(checkpoint, model, optimiser)

    # Set learning rate to zero so the model doesn't change while we're
    # collecting statistics about it.
    for param_group in optimiser.param_groups:
        param_group['lr'] = 0

    # Set up hooks to collect data.
    util.stats.gradient_distribution_hooks(model)

    # Train for one epoch
    train(image_loader, model, criterion, optimiser, epoch, args)
    
    # Output the results.
    with open(log, "w") as f:
        for line in util.stats.get_gradient_stats():
            f.write(line + "\n")


class AverageMeter(object):
    """Compute and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, period, args):
    """Control the learning rate over the epochs."""
    
    if args.use_restarts:
        # Scale learning rate down to 0 according to a cosine curve. Reset to
        # initial value every `args.restart_period` epochs.
        # https://arxiv.org/abs/1608.03983
        epochs_since_restart = epoch % args.restart_period
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epochs_since_restart /
                                           args.restart_period))
    else:
        # Default: decay learning rate by 10x every `period` epochs.
        lr = args.lr * (0.1 ** (epoch // period))
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Compute the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
