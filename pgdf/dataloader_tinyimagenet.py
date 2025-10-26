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
from torchvision.transforms import InterpolationMode
from autoaugment import ImageNetPolicy,CIFAR10Policy
from randaugment import *

transform_none_compose = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
    ]
)

transform_weak_compose = transforms.Compose(
    [
        transforms.RandomCrop(64, padding=4),
        # transforms.RandomResizedCrop(64, interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
    ]
)

transform_strong_compose = transforms.Compose(
    [
        # transforms.RandomResizedCrop(64, interpolation=InterpolationMode.BILINEAR),
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(0.5),
        CIFAR10Policy(),
        #ImageNetPolicy 대신 CIFAR10Policy 사용
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
    ]
)

transform_strong_randaugment_compose = transforms.Compose(
    [
        # transforms.RandomResizedCrop(64, interpolation=InterpolationMode.BILINEAR),
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(0.5),
        # RandAugment(1, 6),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
    ]
)


class Tinyimagenet_dataset(Dataset):
    def __init__(self,r,transform,mode,noise_mode,noise_file="",preaug_file="",pred=[],probability=[],dataset='imagenet200',root_dir="/home/yujin/OpenOOD_baseline/data/images_classic"):
        self.r = r
        self.transform = transform
        self.mode = mode
        self.preaug_file = preaug_file

        if self.mode == "test":
            test_txt = "/home/yujin/OpenOOD_baseline/data/benchmark_imglist/tinyimagenet/test_tin.txt"
            self.samples = []

            with open(test_txt, 'r') as f:
                for line in f:
                    path, label = line.strip().split()
                    self.samples.append((os.path.join(root_dir, path), int(label)))
            self.test_data = [path for path, _ in self.samples]
            self.test_label = [label for _, label in self.samples]

        else:
            # Clean labels
            clean_txt = "/home/yujin/OpenOOD_baseline/data/benchmark_imglist/tinyimagenet/train_tin.txt"
            self.samples = []
            with open(clean_txt, 'r') as f:
                for line in f:
                    path, label = line.strip().split()
                    self.samples.append((os.path.join(root_dir, path), int(label)))  
            self.train_data = [path for path, _ in self.samples]
            self.train_label = [label for _, label in self.samples]
            
            # Noisy labels
            noise_label = []
            with open(noise_file, 'r') as f:
                for line in f:
                    _, label = line.strip().split()
                    noise_label.append(int(label))
            self.noise_label = noise_label
            train_data = np.array(self.train_data)
            
            if self.preaug_file != "":
                all_augmented = torch.load(self.preaug_file)
                train_data = np.concatenate(
                    (
                        train_data,
                        np.array(all_augmented["samples"], dtype=np.uint8).transpose(
                            (0, 2, 3, 1)
                        ),
                    )
                )
                noise_label = np.concatenate(
                    (
                        noise_label,
                        np.array(all_augmented["labels"]),
                    )
                )

            if self.mode == "all":
                # pass
                self.train_data = train_data
                self.noise_label = noise_label
            else:
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]

                elif self.mode == "unlabeled":
                    pred_idx = (1 - pred).nonzero()[0]

                self.train_data = train_data[pred_idx]

                self.noise_label = [noise_label[i] for i in pred_idx]
                print("%s data has a size of %d" % (self.mode, len(self.train_data)))

    def __getitem__(self, index):
        if self.mode == "labeled":
            img_path, target, prob = (
                self.train_data[index],
                self.noise_label[index],
                self.probability[index],
            )
            img = Image.open(img_path).convert('RGB')
            img1 = self.transform[0](img)
            img2 = self.transform[1](img)
            if self.transform[2] == None:
                img3 = img1
                img4 = img2
            else:
                img3 = self.transform[2](img)
                img4 = self.transform[3](img)
            return img1, img2, img3, img4, target, prob
        elif self.mode == "unlabeled":
            img_path = self.train_data[index]
            target = self.noise_label[index]
            img = Image.open(img_path).convert('RGB')
            img1 = self.transform[0](img)
            img2 = self.transform[1](img)
            if self.transform[2] == None:
                img3 = img1
                img4 = img2
            else:
                img3 = self.transform[2](img)
                img4 = self.transform[3](img)
            return img1, img2, img3, img4, target
        elif self.mode == "all":
            img_path, target = self.train_data[index], self.noise_label[index]
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            return img, target, index
        elif self.mode == "test":
            img_path, target = self.test_data[index], self.test_label[index]
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            return img, target

    def __len__(self):
        if self.mode != "test":
            return len(self.train_data)
        else:
            return len(self.test_data)


class Tinyimagenet_dataloader:
    def prob_transform(self,x):
        if random.random() < self.warmup_aug_prob:
            return transform_strong_compose(x)
        else:
            return transform_weak_compose(x)

    def transform_strong(self, x):
        return transform_strong_compose(x)

    def transform_weak(self, x):
        return transform_weak_compose(x)

    def transform_strong_randaugment(self, x):
        return transform_strong_randaugment_compose(x)

    def transform_none(self, x):
        return transform_none_compose(x)

    def __init__(
        self,
        dataset,
        r,
        noise_mode,
        batch_size,
        warmup_batch_size,
        num_workers,
        root_dir,
        noise_file="",
        preaug_file="",
        augmentation_strategy={},
    ):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.warmup_batch_size = warmup_batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.noise_file = noise_file
        self.preaug_file = preaug_file
        self.warmup_aug_prob = augmentation_strategy.warmup_aug_probability
        if "randaugment_params" in augmentation_strategy:
            p = augmentation_strategy["randaugment_params"]
            a = RandAugment(p["n"], p["m"])
            transform_strong_randaugment_compose.transforms.insert(2, a)
        self.transforms = {
            "warmup": self.__getattribute__(augmentation_strategy.warmup_transform),
            "unlabeled": [None for i in range(4)],
            "labeled": [None for i in range(4)],
            "test": None,
        }
        # workaround so it works on both windows and linux
        for i in range(len(augmentation_strategy.unlabeled_transforms)):
            self.transforms["unlabeled"][i] = self.__getattribute__(
                augmentation_strategy.unlabeled_transforms[i]
            )
        for i in range(len(augmentation_strategy.labeled_transforms)):
            self.transforms["labeled"][i] = self.__getattribute__(
                augmentation_strategy.labeled_transforms[i]
            )
        if self.dataset == 'tinyimagenet':
            self.transforms['test'] = transforms.Compose(
                [   
                    transforms.Resize(64, interpolation=InterpolationMode.BILINEAR),
                    transforms.CenterCrop(64),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                ]
            )

        if augmentation_strategy.preaugment and not os.path.exists(self.preaug_file):
            print(f"Preaugmenting and saving to {self.preaug_file}...")
            test_dataset = Tinyimagenet_dataset(
                dataset=self.dataset,
                noise_mode=self.noise_mode,
                noise_file=self.noise_file,
                r=self.r,
                root_dir=self.root_dir,
                transform=self.__getattribute__(
                    augmentation_strategy.preaugment["transform"]
                ),
                mode="all",
            )
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            all_augmented = {"samples": [], "labels": []}
            for i in range(augmentation_strategy.preaugment["ratio"] - 1):
                for img, target, index in test_loader:
                    for j in range(len(img)):
                        all_augmented["samples"].append(img[j].numpy())
                        all_augmented["labels"].append(target[j])
            torch.save(all_augmented, self.preaug_file)

    def run(self, mode, pred=[], prob=[]):
        if mode == "warmup":
            all_dataset = Tinyimagenet_dataset(
                dataset=self.dataset,
                noise_mode=self.noise_mode,
                r=self.r,
                root_dir=self.root_dir,
                transform=self.transforms["warmup"],
                mode="all",
                noise_file=self.noise_file,
                preaug_file=self.preaug_file,
            )
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.warmup_batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )
            return trainloader

        elif mode == "train":
            labeled_dataset = Tinyimagenet_dataset(
                dataset=self.dataset,
                noise_mode=self.noise_mode,
                r=self.r,
                root_dir=self.root_dir,
                transform=self.transforms["labeled"],
                mode="labeled",
                noise_file=self.noise_file,
                pred=pred,
                probability=prob,
                preaug_file=self.preaug_file,
            )
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )

            unlabeled_dataset = Tinyimagenet_dataset(
                dataset=self.dataset,
                noise_mode=self.noise_mode,
                r=self.r,
                root_dir=self.root_dir,
                transform=self.transforms["unlabeled"],
                mode="unlabeled",
                noise_file=self.noise_file,
                pred=pred,
                preaug_file=self.preaug_file,
            )
            if len(unlabeled_dataset) > 0:
                unlabeled_trainloader = DataLoader(
                    dataset=unlabeled_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                )
            else:
                unlabeled_trainloader=None
                
            return labeled_trainloader, unlabeled_trainloader

        elif mode == "test":
            test_dataset = Tinyimagenet_dataset(
                dataset=self.dataset,
                noise_mode=self.noise_mode,
                r=self.r,
                root_dir=self.root_dir,
                transform=self.transforms["test"],
                mode="test",
            )
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            return test_loader

        elif mode == "eval_train":
            eval_dataset = Tinyimagenet_dataset(
                dataset=self.dataset,
                noise_mode=self.noise_mode,
                r=self.r,
                root_dir=self.root_dir,
                transform=self.transforms["test"],
                mode="all",
                noise_file=self.noise_file,
                preaug_file=self.preaug_file,
            )
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            return eval_loader