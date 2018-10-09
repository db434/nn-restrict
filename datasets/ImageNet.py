import os

import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# TODO subclass an abstract Dataset class.
class ImageNet(object):

    _normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    
    # Some sensible defaults.
    name = "ImageNet"
    default_model = "alexnet"
    default_epochs = 90
    default_epoch_period = 30
    location = "/local/scratch/ssd/raw-data"
                                              
    # classes = ???
    
    @staticmethod
    def input_channels():
        return 3
    
    @staticmethod
    def num_classes():
        return 1000

    @staticmethod
    def data_loaders(num_workers, batch_size, distributed=False):
        """Return train and validation data loaders for the ImageNet dataset."""
        return ImageNet.train_loader(num_workers, batch_size, distributed), \
            ImageNet.val_loader(num_workers, batch_size)

    @staticmethod
    def train_loader(num_workers, batch_size, distributed):
        transform = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ImageNet._normalize,
        ])

        dataset = datasets.ImageFolder(os.path.join(ImageNet.location, "train"),
                                       transform)

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
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            ImageNet._normalize,
        ])

        dataset = datasets.ImageFolder(os.path.join(ImageNet.location, "val"),
                                       transform)
            
        loader = data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)
        
        return loader

    @staticmethod
    def test_loader(num_workers, batch_size):
        transform = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            ImageNet._normalize,
        ])

        dataset = datasets.ImageFolder(os.path.join(ImageNet.location, "test"),
                                       transform)
            
        loader = data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)
        
        return loader
