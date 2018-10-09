import os.path
import torch
import torch.nn.functional as F

import locations
import models
import structured
from util import checkpoint, log
from . import trainer


class Trainer(trainer.Trainer):
    """
    Class which trains a model using an experienced teacher model.
    """

    def __init__(self, dataset, model, schedule, args, optimiser=None,
                 criterion=None, teacher=None):
        super(Trainer, self).__init__(dataset, model, schedule, args,
                                      optimiser, criterion)

        if teacher is None:
            self.teacher = self._default_teacher(dataset)
        else:
            self.teacher = teacher

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
        # In some cases it would be quicker to evaluate all teacher outputs in
        # advance, but this can require huge storage for some datasets.
        output = self.model(input_var)
        teacher_output = self.teacher(input_var)

        # TODO get an alpha and a temperature from somewhere. Alpha might
        # even need a schedule.
        alpha = 0.9
        temperature = 20.0
        loss = self.criterion(output, teacher_output, target_var, alpha,
                              temperature)

        return output, loss

    @staticmethod
    def _default_criterion():
        """
        Create a default criterion used to evaluate the loss of the model.

        This version is different from all the criteria in PyTorch because it
        must also receive the teacher's output.

        Based on
          https://arxiv.org/abs/1312.6184
          https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py

        :return: A function which computes the teacher-student loss.
        """

        def kd_loss(student, teacher, target, alpha, temperature):
            """
            Compute the loss in a teacher-student setting.

            :param student: Output from student network.
            :param teacher: Output from teacher network.
            :param target: Target output.
            :param alpha: Weight given to teacher vs target. 1.0 = all loss
            comes from teacher, 0.0 = all loss comes from target.
            :param temperature: Softmax temperature. High = flat
            distributions, low = spiky distributions.
            :return: Loss of student output.
            """

            normal_loss = F.cross_entropy(student, target)

            # Inputs need to be logarithmic, targets need to be linear.
            teacher_loss = F.kl_div(F.log_softmax(student / temperature, dim=1),
                                    F.softmax(teacher / temperature, dim=1))
            teacher_loss *= temperature ** 2

            return (alpha * teacher_loss) + ((1.0 - alpha) * normal_loss)

        return kd_loss

    def _default_teacher(self, dataset):
        """
        Load a model suitable for teaching with the given dataset.

        :param dataset: The dataset to be used for training.
        :return: A model which achieves high accuracy on the dataset.
        """

        # TODO: Offer to train a teacher if not found?
        directory = locations.teachers
        teachers = {
            "CIFAR-10": {
                "name": "resnet56",
                "args": {"width_multiplier": 2},
                "filename": "resnet56-2x-fc.pth.tar"
            },

            "ImageNet": {
                "name": "resnet50",
                "args": {"width_multiplier": 2,
                         "conv_type": structured.depthwise_separable.Conv2d},
                "filename": "resnet50-2x-separable.pth.tar"
            }
        }

        if dataset.name in teachers:
            log.info("Creating teacher network.")
            details = teachers[dataset.name]
            model = models.get_model(details["name"],
                                     distributed=self.distributed,
                                     use_cuda=self.use_cuda, **details["args"])

            # An optimiser is stored with the model, but it isn't needed here.
            # Create a dummy optimiser. (Assuming SGD.)
            optimiser = torch.optim.SGD(model.parameters(), 0)

            # When loading state, data is returned to allow resumption of
            # training. We don't care about that here.
            path = os.path.join(directory, details["filename"])
            _, _ = checkpoint.load_path(path, model, optimiser)

            # Put the teacher into teaching mode: switch off dropout layers,
            # etc. and prevent any weight updates.
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

            return model
        else:
            log.error("No teacher available for", dataset.name, "dataset")
            exit(1)
