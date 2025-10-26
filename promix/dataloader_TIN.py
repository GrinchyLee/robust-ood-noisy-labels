import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from torchvision.transforms import InterpolationMode
import copy
from utils.randaug import *

class TIN_Dataset(Dataset):
    def __init__(self,  image_root, transform, mode, transform_s=None,is_human=True, noise_file='',
                 pred=[], probability=[],probability2=[],log='',r=0.2, dataset='tinyimagenet'):
        self.dataset = dataset
        self.transform = transform
        self.transform_s = transform_s
        self.mode = mode
        self.noise_file = noise_file
        # self.image_root = image_root
        self.r=r
        idx_each_class_noisy = [[] for i in range(200)]
        if mode == 'test':
            test_txt = "/home/yujin/OpenOOD_baseline/data/benchmark_imglist/tinyimagenet/test_tin.txt"

            self.samples = []
            with open(test_txt, 'r') as f:
                for line in f:
                    path, label = line.strip().split()
                    self.samples.append((os.path.join(image_root, path), int(label)))

            self.test_data = [path for path, _ in self.samples]
            self.test_label = [label for _, label in self.samples]
        else:
            self.samples = []
            with open(noise_file, 'r') as f:
                for line in f:
                    path, label = line.strip().split()
                    self.samples.append((os.path.join(image_root, path), int(label)))

            self.train_data = [path for path, _ in self.samples]
            self.train_noisy_labels = [label for _, label in self.samples]
            train_data = self.train_data 
            noise_label = self.train_noisy_labels
            
            clean_txt = "/home/yujin/OpenOOD_baseline/data/benchmark_imglist/tinyimagenet/train_tin.txt"

            # 2. 정답 레이블 로딩
            self.train_labels = []
            with open(clean_txt, 'r') as f:
                for line in f:
                    _, label = line.strip().split()
                    self.train_labels.append(int(label))
            self.noise_or_not = np.transpose(self.train_noisy_labels) != np.transpose(self.train_labels)
            
            if self.mode == 'all_lab':
                self.probability = probability
                self.probability2 = probability2
                self.train_data = train_data 
                self.noise_label = noise_label
            elif self.mode == 'all':
                self.train_data = train_data
                self.noise_label = noise_label
            else:
                if self.mode == 'labeled':
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]
                elif self.mode == 'unlabeled':
                    pred_idx = (1 - pred).nonzero()[0]
                    
                self.train_data = train_data[pred_idx]
                self.noise_label = [noise_label[i] for i in pred_idx]
                self.print_wrapper("%s data has a size of %d" % (self.mode, len(self.noise_label)))
            
    def __getitem__(self, index):
        if self.mode == 'labeled':
            img_path, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
            img = Image.open(img_path).convert('RGB')
            img1 = self.transform(img)
            img2 = self.transform_s(img)
            return img1, img2, target, prob
        elif self.mode == 'unlabeled':
            img_path = self.train_data[index]
            img = Image.open(img_path).convert('RGB')
            img1 = self.transform(img)
            img2 = self.transform_s(img)
            return img1, img2
        elif self.mode == 'all_lab':
            img_path, target, prob, prob2 = self.train_data[index], self.noise_label[index], self.probability[index],self.probability2[index]
            true_labels = self.train_labels[index]
            img = Image.open(img_path).convert('RGB')
            img1 = self.transform(img)
            img2 = self.transform_s(img)
            return img1, img2, target, prob,prob2,true_labels, index
        elif self.mode == 'all':
            img_path, target = self.train_data[index], self.noise_label[index]
            img = Image.open(img_path).convert('RGB')
            if self.transform_s is not None:
                img1 = self.transform(img)
                img2 = self.transform_s(img)
                return img1, img2, target, index
            else:
                img = self.transform(img)
                return img, target, index
        elif self.mode == 'all2':
            img_path, target = self.train_data[index], self.noise_label[index]
            img = Image.open(img_path).convert('RGB')
            img1 = self.transform(img)
            img2 = self.transform_s(img)
            return img1, img2, target, index

        elif self.mode == 'test':
            img_path, target = self.test_data[index], self.test_label[index]
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            return img, target

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")


    def __len__(self):
        if self.mode != 'test':
            return len(self.train_data)
        else:
            return len(self.test_data)


class TIN_dataloader():
    def __init__(self,args,log):
        self.image_root = os.path.join(args.data_path, 'images_classic')
        self.log = log
        self.batch_size = args.batch_size
        self.num_workers = 8
        self.noise_file = args.noise_file
        self.dataset = args.dataset
        self.is_human = args.is_human
        # transforms
        self.r = args.noise_rate
        self.transform_train = transforms.Compose([
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.transform_train_s = copy.deepcopy(self.transform_train)
        self.transform_train_s.transforms.insert(0, RandomAugment(3,5))
        self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
        ])
        # true clean labels
        with open(self.noise_file if self.noise_file else os.path.join(args.data_path, "benchmark_imglist/tinyimagenet/train_tin.txt"), "r") as f:
            self.true_labels = [int(line.strip().split()[1]) for line in f]

        # load noise if specified
        self.noise_labels = self.true_labels
        self.noise_or_not = [False] * len(self.true_labels)

    def run(self, mode, pred=None, prob=None, prob2=None):
        if mode == 'warmup':
            all_dataset = TIN_Dataset(dataset=self.dataset, image_root = self.image_root, transform=self.transform_train,transform_s=self.transform_train_s,
                                         mode = 'all', r = self.r, noise_file=self.noise_file)
            trainloader = DataLoader(dataset=all_dataset, batch_size = self.batch_size,shuffle=True,num_workers=self.num_workers)
            return trainloader, all_dataset.train_noisy_labels
        
        elif mode =='train':
            labeled_dataset = TIN_Dataset(dataset=self.dataset, image_root = self.image_root, transform=self.transform_train,transform_s=self.transform_train_s,
                                         mode = 'all_lab', r = self.r, noise_file=self.noise_file,pred=pred, probability=prob,probability2=prob2, log=self.log)
            labeled_trainloader = DataLoader(dataset=labeled_dataset, batch_size = self.batch_size,shuffle=True,num_workers=self.num_workers,pin_memory=True,drop_last=True)
            return labeled_trainloader, labeled_dataset.train_noisy_labels
        
        elif mode == 'test':
            test_dataset = TIN_Dataset(dataset=self.dataset, is_human = self.is_human, transform = self.transform_test, r=self.r,mode='test',image_root=self.image_root)
            test_loader = DataLoader(dataset = test_dataset,batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers)
            return test_loader
        
        elif mode == 'eval_train':
            eval_dataset = TIN_Dataset(dataset=self.dataset, is_human=self.is_human, transform=self.transform_test, 
                                            mode='all', noise_file=self.noise_file, r=self.r, 
                                            image_root=self.image_root, transform_s=None)
            eval_loader = DataLoader(dataset=eval_dataset, batch_size=self.batch_size, 
                                    shuffle=False, num_workers=self.num_workers)

            return eval_loader, eval_dataset.noise_or_not