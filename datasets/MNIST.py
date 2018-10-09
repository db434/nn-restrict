import torch.utils.data as data
import torchvision.datasets
import torchvision.transforms as transforms

import locations


# TODO subclass an abstract Dataset class
class MNIST(object):

    _normalize = transforms.Normalize((0.1307,), (0.3081,))

    # Some sensible defaults.
    name = "MNIST"
    default_model = "lenet5"
    location = locations.mnist

    # See training.lr_schedule.py for explanation.
    default_lr = 0.05
    default_lr_steps = [(10, 0.1)]
    default_epochs = 20

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    @staticmethod
    def input_channels():
        return 1

    @staticmethod
    def num_classes():
        return len(MNIST.classes)

    @staticmethod
    def data_loaders(num_workers, batch_size, distributed=False):
        """Return train and validation data loaders for the MNIST dataset."""
        return MNIST.train_loader(num_workers, batch_size, distributed), \
            MNIST.val_loader(num_workers, batch_size)

    @staticmethod
    def train_loader(num_workers, batch_size, distributed):
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            MNIST._normalize,
        ])

        dataset = torchvision.datasets.MNIST(root=MNIST.location,
                                             train=True, download=True,
                                             transform=transform)

        if distributed:
            sampler = data.distributed.DistributedSampler(dataset)
        else:
            sampler = None

        loader = data.DataLoader(
            dataset, batch_size=batch_size, shuffle=(sampler is None),
            num_workers=num_workers, pin_memory=True, sampler=sampler)

        return loader

    @staticmethod
    def val_loader(num_workers, batch_size):
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            MNIST._normalize,
        ])

        dataset = torchvision.datasets.MNIST(root=MNIST.location,
                                             train=False, download=True,
                                             transform=transform)

        loader = data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)

        return loader
