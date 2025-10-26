import json
import os
import pickle
import random
import _pickle as cPickle
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset



def unpickle(file):
    with open(file, "rb") as fo:
        return cPickle.load(fo, encoding="latin1")

transform_weak_10_compose = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)


transform_weak_100_compose = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ]
)

class cifar_dataset(Dataset):
    def __init__(self,dataset,train_data,r,noise_mode,train_label,transform,mode,noise_file="",preaug_file="",):
        self.r = r
        self.transform = transform
        self.train_data = train_data
        self.train_label = train_label.tolist()
        self.mode = mode
        print(self.r)
        
        noise_label = train_label.tolist()
        self.noise_label = noise_label
        pickle.dump(noise_label, open(noise_file, "wb"))


    def __getitem__(self, index):
        if self.mode == "all": #for D_a
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target, index
        elif self.mode == "clean": #for D_e
            img, target = self.train_data[index], self.train_label[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2, target

    def __len__(self):
        if self.mode != "test":
            return len(self.train_data)
        else:
            return len(self.test_data)


class easy_dataloader:
    def transform_weak_100(self, x):
        return transform_weak_100_compose(x)

    def transform_weak_10(self, x):
        return transform_weak_10_compose(x)
    

    def __init__(self,dataset,r,noise_mode,batch_size,warmup_batch_size,num_workers,root_dir,noise_file="",preaug_file="",augmentation_strategy={},):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.warmup_batch_size = warmup_batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.noise_file = noise_file
        self.transforms = {
            "warmup": self.__getattribute__(augmentation_strategy.warmup_transform)
        }
    
    def run(self, mode, train_data, train_label):
        if mode == "warmup": # for D_a
            all_dataset = cifar_dataset(
                dataset=self.dataset,
                noise_mode=self.noise_mode,
                r=self.r,
                transform=self.transforms["warmup"],
                mode="all",
                noise_file=self.noise_file,
                train_data=train_data,
                train_label=train_label,
            )
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.warmup_batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )
            return trainloader

        if mode == "clean": # for D_e
            all_dataset = cifar_dataset(
                dataset=self.dataset,
                noise_mode=self.noise_mode,
                r=self.r,
                transform=self.transforms["warmup"],
                mode="clean",
                noise_file=self.noise_file,
                train_data=train_data,
                train_label=train_label,
            )
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.warmup_batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )
            return trainloader
