import torch
import torch.nn as nn

from . import functional


class Quantisable(nn.Module):
    """
    Module which quantises its weights.

    A full-precision copy of weights is retained to allow gradients to
    accumulate over multiple small steps.
    """
    def __init__(self, module):
        super(Quantisable, self).__init__()
        self.module = module
        self.weight_transform = None
        self.backup_parameters = {}

    def set_weight_transform(self, transform_fn):
        self.weight_transform = transform_fn

    def quantise_parameters(self):
        """For all parameters contained within this module, store a backup
        copy, and quantise those used for computation."""
        if self.weight_transform is not None:
            # Store full-precision backups of all parameters, if they don't
            # already exist.
            if len(self.backup_parameters) == 0:
                for name, tensor in self.module.state_dict().items():
                    # Creating a clone means that gradients will reach both the
                    # quantised and full-precision versions of the tensor.
                    # If weights are not to be stored in full-precision,
                    # quantise here.
                    self.backup_parameters[name] = tensor
                    self.module.state_dict()[name] = tensor.clone()

            for name, tensor in self.module.state_dict().items():
                # Newer versions of PyTorch include the epoch counter as a
                # parameter. We don't want to modify that!
                if tensor.dtype == torch.long:
                    continue

                quantised = tensor
                full_precision = self.backup_parameters[name]
                assert full_precision.size() == quantised.size()

                # Bypass the Variable interface so PyTorch doesn't get
                # confused by the Tensor contents changing. (Hack)
                quantised.data.copy_(functional.quantise(full_precision,
                                                         self.weight_transform))

    def restore_parameters(self):
        if len(self.backup_parameters) > 0:
            for name, tensor in self.module.state_dict().items():
                quantised = tensor
                full_precision = self.backup_parameters[name]
                assert full_precision.size() == quantised.size()

                # Bypass the Variable interface so PyTorch doesn't get
                # confused by the Tensor contents changing. (Hack)
                quantised.data.copy_(full_precision)

    def forward(self, *inputs, **kwargs):
        self.quantise_parameters()
        # Could do something similar to all input tensors. Might not be
        # particularly useful if unquantised activations are then passed
        # between submodules...

        result = self.module(*inputs, **kwargs)

        # It would be nice to restore the full-precision parameters here,
        # but then there is no obvious way to use the quantised versions in
        # the backward pass. Instead, I leave the quantised versions in
        # place, and only restore full-precision parameters when storing the
        # model.

        return result


class Quantiser(nn.Module):
    """
    Module which quantises its given input data.
    """

    def __init__(self, quantisation_fn=None):
        """
        Module constructor.

        :param quantisation_fn: A function which takes a tensor as input and
        returns a transformed tensor.
        """
        super(Quantiser, self).__init__()
        self.quantisation_fn = quantisation_fn

    def set_quantisation(self, quantisation_fn):
        """
        Change the quantisation function post-initialisation.

        :param quantisation_fn: A function which takes a tensor as input and
        returns a transformed tensor.
        """
        self.quantisation_fn = quantisation_fn

    def forward(self, x):
        return functional.quantise(x, self.quantisation_fn)
