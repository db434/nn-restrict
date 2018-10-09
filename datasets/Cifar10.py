import torch.utils.data as data
import torchvision.datasets
import torchvision.transforms as transforms


# TODO subclass an abstract Dataset class.
class Cifar10(object):

    _normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                      std=[0.2023, 0.1994, 0.2010])
    
    # Some sensible defaults.
    name = "CIFAR-10"
    default_model = "aaronnet"
    default_epochs = 200
    default_epoch_period = 80
    location = "/local/scratch/ssd/cifar10"
                                              
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
               'ship', 'truck')
    
    @staticmethod
    def input_channels():
        return 3
    
    @staticmethod
    def num_classes():
        return len(Cifar10.classes)

    @staticmethod
    def data_loaders(num_workers, batch_size, distributed=False):
        """Return train and validation data loaders for the CIFAR-10 dataset."""
        return Cifar10.train_loader(num_workers, batch_size, distributed), \
            Cifar10.val_loader(num_workers, batch_size)

    @staticmethod
    def train_loader(num_workers, batch_size, distributed):
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            Cifar10._normalize,
        ])

        dataset = torchvision.datasets.CIFAR10(root=Cifar10.location,
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
            transforms.ToTensor(),
            Cifar10._normalize,
        ])

        dataset = torchvision.datasets.CIFAR10(root=Cifar10.location,
                                               train=False, download=True,
                                               transform=transform)
            
        loader = data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)
        
        return loader
