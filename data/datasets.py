import numpy as np 
import torchvision.transforms as transforms
from .cifar import CIFAR10, CIFAR100
from.Imagenet200 import ImageNet200
from torchvision.transforms import InterpolationMode



train_cifar10_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4), 
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_cifar10_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_cifar100_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

test_cifar100_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

train_imagenet200_transform = transforms.Compose([
            transforms.RandomResizedCrop(128, interpolation=InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])
test_imagenet200_transform = transforms.Compose([
            transforms.Resize((128, 128), interpolation=InterpolationMode.BILINEAR),  # 크기 고정
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])


def input_dataset(dataset, noise_type, noise_path, is_human,noise_file=None):
    if dataset == 'cifar10':
        train_dataset = CIFAR10(root='~/data/',
                                download=True,  
                                train=True, 
                                transform = train_cifar10_transform,
                                noise_type = noise_type,
                                noise_path = noise_path, is_human=is_human
                           )
        test_dataset = CIFAR10(root='~/data/',
                                download=False,  
                                train=False, 
                                transform = test_cifar10_transform,
                                noise_type=noise_type
                          )
        num_classes = 10
        num_training_samples = 50000
    elif dataset == 'cifar100':
        train_dataset = CIFAR100(root='~/data/',
                                download=True,  
                                train=True, 
                                transform=train_cifar100_transform,
                                noise_type=noise_type,
                                noise_path = noise_path, is_human=is_human
                            )
        test_dataset = CIFAR100(root='~/data/',
                                download=False,  
                                train=False, 
                                transform=test_cifar100_transform,
                                noise_type=noise_type
                            )
        num_classes = 100
        num_training_samples = 50000
        
    elif dataset == 'imagenet200':
        train_dataset = ImageNet200(
            train=True,
            transform=train_imagenet200_transform,
            noise_file=noise_file,
            is_human=is_human
        )
        test_dataset = ImageNet200(
            train=False,
            transform=test_imagenet200_transform,
            noise_file=noise_file,
            is_human=is_human
        )
        num_classes = 200
        num_training_samples = len(train_dataset) 
             
    return train_dataset, test_dataset, num_classes, num_training_samples









