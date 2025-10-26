import sys

import numpy as np
from PIL import Image
import torchvision
from torch.utils.data.dataset import Subset
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances 
import torch
import torch.nn.functional as F
import random 
import os
import json
from numpy.testing import assert_array_almost_equal
from torch.utils.data import Dataset

def get_TIN(root, cfg_trainer, train=True,
                    transform_train=None, transform_train_aug=None, transform_val=None,
                    download=False, noise_file=''):
    
    txt_dir = "/home/dm/Leeyujin/OpenOOD/data/benchmark_imglist/tinyimagenet"
    test_txt = os.path.join(txt_dir, 'test_tin.txt')
    
    if train:
        train_txt = os.path.join(txt_dir, 'train_tin.txt')
        val_txt = os.path.join(txt_dir, 'val_tin.txt')
        train_dataset = Tinyimagenet_train(train_txt, root, cfg_trainer,
                                          transform=transform_train)
        val_dataset = Tinyimagenet_val(val_txt, root, cfg_trainer,
                                      transform=transform_val)
        if cfg_trainer.get("noise_file", ""):
            print("[Skip] Skipping noise injection; loading from file instead.")
            
        elif cfg_trainer.get('asym', False):
            train_dataset.asymmetric_noise()
            val_dataset.asymmetric_noise()
        else:
            train_dataset.symmetric_noise()
            val_dataset.symmetric_noise()
        print(f"Train: {len(train_dataset)} Val: {len(val_dataset)}")
    else:
        train_dataset = []
        val_dataset = Tinyimagenet_val(test_txt, root, cfg_trainer,
                                      transform=transform_val)
        print(f"Test: {len(val_dataset)}")
    return train_dataset, val_dataset



class Tinyimagenet_train(Dataset):
    def __init__(self, txt_file, root, cfg_trainer, transform=None, target_transform=None):
        self.root = root
        self.cfg_trainer = cfg_trainer
        self.transform = transform
        self.num_classes = 200
        self.target_transform = target_transform
        self.image_paths, self.train_labels = self._load_txt(txt_file)
        self.train_labels = np.array(self.train_labels)
        self.train_labels_gt = self.train_labels.copy()

        self.noise_indx = []

        if self.cfg_trainer.get("noise_file"):
            noise_file = self.cfg_trainer["noise_file"]
            print(f"[Load] Loading noisy labels from: {noise_file}")

            # 상대경로 기준 매핑
            noisy_label_dict = {}
            with open(noise_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        path, noisy_label = parts
                        noisy_label_dict[path] = int(noisy_label)

            noisy_count = 0
            for i in range(len(self.image_paths)):
                rel_path = os.path.relpath(self.image_paths[i], self.root)  # ✅ 상대경로로 맞춰줌
                if rel_path in noisy_label_dict:
                    self.train_labels[i] = noisy_label_dict[rel_path]
                    if self.train_labels[i] != self.train_labels_gt[i]:
                        noisy_count += 1
                else:
                    raise ValueError(f"Path not found in noisy file: {rel_path}")

            noise_ratio = noisy_count / len(self.image_paths)
            print(f"[Applied] Loaded noisy labels from file. Noise ratio: {noise_ratio:.2%} ({noisy_count}/{len(self.image_paths)})")

        
    def _load_txt(self, txt_file):
        image_paths = []
        labels = []
        with open(txt_file, 'r') as f:
            for line in f:
                path, label = line.strip().split()
                image_paths.append(os.path.join(self.root, path))
                labels.append(int(label))
        return image_paths, labels
        
    def symmetric_noise(self):
        indices = np.random.permutation(len(self.image_paths))
        for i, idx in enumerate(indices):
            if i < self.cfg_trainer['percent'] * len(self.image_paths):
                self.noise_indx.append(idx)
                self.train_labels[idx] = np.random.randint(self.num_classes)
        noisy_count = np.sum(self.train_labels != self.train_labels_gt)
        noise_ratio = noisy_count / len(self.train_labels)
        print(f"[Check] Symmetric noise inserted: {noisy_count}/{len(self.train_labels)} samples ({noise_ratio:.2%})")

    def multiclass_noisify(self, y, P, random_state=0):
        """ Flip classes according to transition probability matrix T.
        It expects a number between 0 and the number of classes - 1.
        """

        assert P.shape[0] == P.shape[1]
        assert np.max(y) < P.shape[0]

        # row stochastic matrix
        assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
        assert (P >= 0.0).all()

        m = y.shape[0]
        new_y = y.copy()
        flipper = np.random.RandomState(random_state)

        for idx in np.arange(m):
            i = y[idx]
            # draw a vector with only an 1
            flipped = flipper.multinomial(1, P[i, :], 1)[0]
            new_y[idx] = np.where(flipped == 1)[0]

        return new_y



    def build_for_imagenet(self, size, noise):
        P = np.eye(size)
        cls1, cls2 = np.random.choice(size, size=2, replace=False)
        P[cls1, cls2] = noise
        P[cls2, cls1] = noise
        P[cls1, cls1] = 1.0 - noise
        P[cls2, cls2] = 1.0 - noise
        assert_array_almost_equal(P.sum(axis=1), 1, 1)
        return P


    def asymmetric_noise(self, asym=False, random_shuffle=False):
        P = np.eye(self.num_classes)
        n = self.cfg_trainer['percent']
        nb_superclasses = 20
        nb_subclasses = 10

        if n > 0.0:
            for i in np.arange(nb_superclasses):
                init, end = i * nb_subclasses, (i+1) * nb_subclasses
                P[init:end, init:end] = self.build_for_cifar100(nb_subclasses, n)

            y_train_noisy = self.multiclass_noisify(self.train_labels, P=P,
                                               random_state=0)
            actual_noise = (y_train_noisy != self.train_labels).mean()
            assert actual_noise > 0.0
            self.train_labels = y_train_noisy

            

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_path = self.image_paths[index]
        target = self.train_labels[index]
        target_gt = self.train_labels_gt[index]
        
        img = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index, target_gt

    def __len__(self):
        return len(self.train_data)

    def __len__(self):
        return len(self.image_paths)


class Tinyimagenet_val(Dataset):
    def __init__(self, txt_file, root, cfg_trainer, transform=None,target_transform=None,):
        self.root = root
        self.cfg_trainer = cfg_trainer
        self.transform = transform
        self.num_classes = 200
        self.target_transform = target_transform
        self.image_paths, self.train_labels = self._load_txt(txt_file)
        self.train_labels = np.array(self.train_labels)
        self.train_labels_gt = self.train_labels.copy()

    def _load_txt(self, txt_file):
        image_paths, labels = [], []
        with open(txt_file, 'r') as f:
            for line in f:
                path, label = line.strip().split()
                image_paths.append(os.path.join(self.root, path))
                labels.append(int(label))
        return image_paths, labels
    
    def symmetric_noise(self):
        indices = np.random.permutation(len(self.image_paths))
        for i, idx in enumerate(indices):
            if i < self.cfg_trainer['percent'] * len(self.image_paths):
                self.train_labels[idx] = np.random.randint(self.num_classes)
        noisy_count = np.sum(self.train_labels != self.train_labels_gt)
        noise_ratio = noisy_count / len(self.train_labels)
        print(f"[Check] Symmetric noise inserted: {noisy_count}/{len(self.train_labels)} samples ({noise_ratio:.2%})")
        
        # percent_str = str(self.cfg_trainer['percent'])
        # save_dir = "/home/young/Leeyujin/SOP_no_val/data_loader/noisy_dataset"
        # os.makedirs(save_dir, exist_ok=True)
        # save_path = os.path.join(save_dir, f"train_imagenet200_sym_{percent_str}.txt")

        # with open(save_path, 'w') as f:
        #     for i in range(len(self.image_paths)):
        #         path = self.image_paths[i]
        #         true_label = self.train_labels_gt[i]
        #         noisy_label = self.train_labels[i]
        #         f.write(f"{path} {true_label} {noisy_label}\n")

        # print(f"[Saved] Noisy label info saved to: {save_path}")

    def multiclass_noisify(self, y, P, random_state=0):
        """ Flip classes according to transition probability matrix T.
        It expects a number between 0 and the number of classes - 1.
        """

        assert P.shape[0] == P.shape[1]
        assert np.max(y) < P.shape[0]

        # row stochastic matrix
        assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
        assert (P >= 0.0).all()

        m = y.shape[0]
        new_y = y.copy()
        flipper = np.random.RandomState(random_state)

        for idx in np.arange(m):
            i = y[idx]
            # draw a vector with only an 1
            flipped = flipper.multinomial(1, P[i, :], 1)[0]
            new_y[idx] = np.where(flipped == 1)[0]

        return new_y

    def build_for_imagenet(self, size, noise):
        P = np.eye(size)
        cls1, cls2 = np.random.choice(size, size=2, replace=False)
        P[cls1, cls2] = noise
        P[cls2, cls1] = noise
        P[cls1, cls1] = 1.0 - noise
        P[cls2, cls2] = 1.0 - noise
        assert_array_almost_equal(P.sum(axis=1), 1, 1)
        return P


    def asymmetric_noise(self, asym=False, random_shuffle=False):
        P = np.eye(self.num_classes)
        n = self.cfg_trainer['percent']
        nb_superclasses = 20
        nb_subclasses = 10

        if n > 0.0:
            for i in np.arange(nb_superclasses):
                init, end = i * nb_subclasses, (i+1) * nb_subclasses
                P[init:end, init:end] = self.build_for_imagenet(nb_subclasses, n)

            y_train_noisy = self.multiclass_noisify(self.train_labels, P=P,
                                               random_state=0)
            actual_noise = (y_train_noisy != self.train_labels).mean()
            assert actual_noise > 0.0
            self.train_labels = y_train_noisy



    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_path = self.image_paths[index]
        target = self.train_labels[index]
        target_gt = self.train_labels_gt[index]
        
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index, target_gt