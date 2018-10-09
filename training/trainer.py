import time
import torch

from structured import convert_to_conv as c2c
from util import log


class Trainer(object):
    """Class which takes a model and trains it."""

    def __init__(self, dataset, model, schedule, args, optimiser=None,
                 criterion=None):
        self.dataset = dataset
        self.train_loader, self.val_loader = dataset.data_loaders(
            args.workers, args.batch_size, distributed=args.distributed)

        self.model = model
        self.lr_schedule = schedule

        # Extract some relevant arguments.
        self.distributed = args.distributed
        self.print_frequency = args.print_freq
        self.use_cuda = args.cuda

        # Recurrent models need extra care taken with their gradients to avoid
        # zeros and infinities.
        self.use_grad_clipping = False
        for module in self.model.modules():
            if isinstance(module, torch.nn.modules.rnn.RNNBase) or \
                    isinstance(module, c2c.RNNBase):
                self.use_grad_clipping = True
                break

        if optimiser is None:
            self.optimiser = self._default_optimiser(args, model)
        else:
            self.optimiser = optimiser

        if criterion is None:
            self.criterion = self._default_criterion()
        else:
            self.criterion = criterion

    def train_epoch(self, epoch, data_loader=None):
        """
        Train for one epoch.

        :param epoch: The current epoch number.
        :param data_loader: (Optional) torch.data.DataLoader to provide input.
        :return: Loss, top-1 and top-5 accuracies for this epoch.
        """

        if data_loader is None:
            data_loader = self.train_loader
        if self.distributed:
            data_loader.sampler.set_epoch(epoch)

        self.set_learning_rate(self.lr_schedule.get_learning_rate(epoch))

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # Switch to train mode
        self.model.train()

        end = time.time()
        for i, (data, target) in enumerate(data_loader):
            data_time.update(time.time() - end)

            if self.use_cuda:
                target = target.cuda(async=True)

            # Compute output
            output, loss = self.minibatch(data, target)

            # Measure accuracy and record loss
            prec1, prec5 = self.accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data.item(), data.size(0))
            top1.update(prec1.item(), data.size(0))
            top5.update(prec5.item(), data.size(0))

            # Compute gradients.
            self.optimiser.zero_grad()
            loss.backward()

            # 0.25 is the default value from here:
            # https://github.com/pytorch/examples/blob/master/word_language_model/main.py
            if self.use_grad_clipping:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)

            # Update model.
            self.optimiser.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.print_frequency == 0:
                log.info(
                    'Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        epoch, i, len(data_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, top5=top5))

        return losses.avg, top1.avg, top5.avg

    def validate(self, data_loader=None):
        """
        Run the validation dataset through the model.

        :param data_loader: (Optional) torch.data.DataLoader to provide input.
        :return: Loss, top-1 and top-5 accuracies for the dataset.
        """

        if data_loader is None:
            data_loader = self.val_loader

        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # Switch to evaluate mode
        self.model.eval()

        end = time.time()
        for i, (data, target) in enumerate(data_loader):
            if self.use_cuda:
                target = target.cuda(async=True)

            # Compute output
            output, loss = self.minibatch(data, target)

            # Measure accuracy and record loss
            prec1, prec5 = self.accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data.item(), data.size(0))
            top1.update(prec1.item(), data.size(0))
            top5.update(prec5.item(), data.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.print_frequency == 0:
                log.info(
                    'Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(data_loader), batch_time=batch_time,
                        loss=losses, top1=top1, top5=top5))

        log.info(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                 .format(top1=top1, top5=top5))

        return losses.avg, top1.avg, top5.avg

    def minibatch(self, input_data, target):
        """
        Pass one minibatch of data through the model.

        :param input_data: Tensor of input data.
        :param target: Ground truth output data.
        :return: Output produced by the model and the loss.
        """
        input_var = torch.autograd.Variable(input_data)
        target_var = torch.autograd.Variable(target)

        # Compute output
        output = self.model(input_var)
        loss = self.criterion(output, target_var)

        return output, loss

    def set_learning_rate(self, lr):
        """
        Update the learning rate.

        :param lr: New learning rate.
        """
        for param_group in self.optimiser.param_groups:
            param_group['lr'] = lr

    @staticmethod
    def accuracy(output, target, topk=(1,)):
        """
        Compute the top-k precision for the given values of k.

        :param output: Output produced by model.
        :param target: Ground truth output.
        :param topk: Iterable containing all values of k.
        :return: List of top-k precisions in the same order as `topk` input.
        """
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

    @staticmethod
    def _default_optimiser(args, model):
        """
        Create a default optimiser to update the model's weights.

        :param args: Command line parameters.
        :param model: Model to be optimised.
        :return: A torch.optim.Optimizer which will update the model.
        """
#        return torch.optim.Adam(model.parameters(), args.lr)
        return torch.optim.SGD(model.parameters(), args.lr,
                               momentum=args.momentum,
                               weight_decay=args.weight_decay)

    @staticmethod
    def _default_criterion():
        """
        Create a default criterion used to evaluate the loss of the model.

        :return: A function which takes the model's output and the ground
        truth and generates a loss value.
        """
        return torch.nn.functional.cross_entropy


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
