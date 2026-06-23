from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch
import torch.utils.data as data
from .utils import download_url, check_integrity, multiclass_noisify

class ImageNet200(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None,
                 noise_file=None, is_human=True,train=True):
        self.transform = transform
        self.target_transform = target_transform
        self.dataset = 'imagenet200'
        self.nb_classes = 200
        self.noise_file = noise_file
        self.is_human = is_human
        self.train = train
        image_root = "/home/yujin/OpenOOD_baseline/data/images_largescale"
        clean_image_path = "/home/yujin/OpenOOD_baseline/data/benchmark_imglist/imagenet200/train_imagenet200.txt"
        
        if self.train:
            # 1. 클린 라벨 로드 (원래 라벨)
            self.train_data_path = []
            self.train_labels = []
            with open(clean_image_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    image_rel_path, label = line.split()
                    full_path = os.path.join(image_root, image_rel_path)
                    self.train_data_path.append(full_path)
                    self.train_labels.append(int(label))
                    

            # 2. 노이즈 라벨 로드 (노이즈 실험을 위한 별도 파일)
            noisy_labels = []
            if noise_file is None:
                raise ValueError("noise_file must be provided for training with noise labels.")
            with open(noise_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    _, label = line.split()  # 이미지 경로는 동일하다고 가정
                    noisy_labels.append(int(label))

            self.train_noisy_labels = noisy_labels
            self.noise_or_not = [clean != noisy for clean, noisy in zip(self.train_labels, self.train_noisy_labels)]
            self.actual_noise_rate = sum(self.noise_or_not) / len(self.noise_or_not)
            print(f"Overall noise rate: {self.actual_noise_rate:.4f}")
        else:
            text_file = "/home/yujin/OpenOOD_baseline/data/benchmark_imglist/imagenet200/test_imagenet200.txt"
            self.test_data_path = []
            self.test_labels = []
            with open(text_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    image_rel_path, label = line.split()
                    full_path = os.path.join(image_root, image_rel_path)
                    self.test_data_path.append(full_path)
                    self.test_labels.append(int(label))

            print(f"Loaded {len(self.test_data_path)} test samples.")

    def __getitem__(self, index):
        if self.train:
            img_path, label = self.train_data_path[index], self.train_noisy_labels[index]
        else:
            img_path,label = self.test_data_path[index],self.test_labels[index]

        # 실제 이미지 로딩
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label, index
    
    def __len__(self):
        if self.train:
            return len(self.train_data_path)
        else:
            return len(self.test_data_path)
        
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str