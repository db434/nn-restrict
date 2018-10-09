import abc
import math


# TODO: These classes are very similar (but perhaps more general) than those in
# torch.optim.lr_scheduler. Consider using those instead.
class LRSchedule(object):
    """Base class for all learning rate schedules. Not to be used directly."""

    def __init__(self):
        return

    @abc.abstractmethod
    def get_learning_rate(self, epoch):
        """
        Compute the learning rate to be used at the given epoch.

        :param epoch: The current epoch.
        :return: The learning rate to use.
        """
        return


class StepSchedule(LRSchedule):
    """A schedule where the learning rate remains constant for multiple
    epochs at a time."""

    def __init__(self, initial_lr, steps):
        """
        Create a schedule.

        :param initial_lr: Learning rate at epoch 0.
        :param steps: List of pairs telling when to drop the learning rate
        and by how much. Use (epoch, factor), where a factor of 0.1 means
        dividing the learning rate by 10.
        """
        super(StepSchedule, self).__init__()
        self.initial_lr = initial_lr
        self.steps = steps

    def get_learning_rate(self, epoch):
        lr = self.initial_lr

        for (drop_epoch, factor) in self.steps:
            if epoch >= drop_epoch:
                lr *= factor
            else:
                break

        return lr


class LinearSchedule(LRSchedule):
    """A schedule where the learning rate changes by a constant amount each
    epoch."""

    def __init__(self, initial_lr, final_lr, epochs):
        """
        Create a schedule.

        :param initial_lr: Learning rate at epoch 0.
        :param final_lr: Learning rate in final epoch.
        :param epochs: Total number of epochs.
        """
        super(LinearSchedule, self).__init__()
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.epochs = epochs

    def get_learning_rate(self, epoch):
        return self.initial_lr + (epoch / self.epochs) * \
                                 (self.final_lr - self.initial_lr)


class CosineRestartSchedule(LRSchedule):
    """A schedule which continuously reduces the learning rate following a
    cosine curve, but returns instantly to the initial learning rate instead
    of following the cosine curve back up.

    Based on https://arxiv.org/abs/1608.03983"""

    def __init__(self, initial_lr, period):
        """
        Create a schedule.

        :param initial_lr: Learning rate at epoch 0.
        :param period: Number of epochs taken to reduce learning rate to
        zero. The learning rate will return to the initial value at the next
        step.
        """
        super(CosineRestartSchedule, self).__init__()
        self.initial_lr = initial_lr
        self.period = period

    def get_learning_rate(self, epoch):
        epochs = epoch % self.period

        # cos has a range of [-1, 1].
        # Add 1 so the whole range is positive: [0, 2].
        # Multiply by 0.5 to create a scaling factor in [0, 1].
        lr = 0.5 * self.initial_lr * (1 + math.cos(math.pi * epochs /
                                                   self.period))
        return lr
