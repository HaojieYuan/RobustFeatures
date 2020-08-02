from visualize import save_img_tensor
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
import torch
import numpy as np
from PIL import Image
from DCT import split_single_img_tensor
import random
import pdb

class udf_dataset_with_name(Dataset):
    ''' with data prefix and data list, return CIFAR-10 dataset. '''
    def __init__(self, prefix, list_file, transform=None, frequency='normal'):
        self.prefix = prefix
        self.imgpaths = []
        self.labels = []
        self.names = []

        with open(list_file) as f:
            for line in f:
                self.imgpaths.append(os.path.join(prefix, line.strip()))
                self.labels.append(int(line.strip().split('/')[1]))
                self.names.append(line.strip())

        self.num = len(self.imgpaths)
        self.transform = transform
        self.frequency = frequency

    def __len__(self):
        return self.num

    def random_high(self):
        idx = random.randint(0, self.num-1)
        img = Image.open(self.imgpaths[idx])
        img = self.transform(img)
        label = self.labels[idx]
        img, _ = split_single_img_tensor(img)

        return img

    def random_low(self):
        idx = random.randint(0, self.num-1)
        img = Image.open(self.imgpaths[idx])
        img = self.transform(img)
        label = self.labels[idx]
        _, img = split_single_img_tensor(img)

        return img

    def __getitem__(self, idx):
        img = Image.open(self.imgpaths[idx])
        img = self.transform(img)
        label = self.labels[idx]

        if self.frequency == 'normal':
            img = img
        elif self.frequency == 'high':
            img, _ = split_single_img_tensor(img)
            img = img + self.random_low()
        elif self.frequency == 'low':
            _, img = split_single_img_tensor(img)
            img = img + self.random_high()

        return img, label, self.names[idx]


transform = transforms.Compose([transforms.ToTensor()])

high_dataset = udf_dataset_with_name('/home/haojieyuan/Data/CIFAR_10_data/images/train',
                           '/home/haojieyuan/Data/CIFAR_10_data/images/train.txt',
                           transform=transform, frequency='high')

low_dataset = udf_dataset_with_name('/home/haojieyuan/Data/CIFAR_10_data/images/train',
                           '/home/haojieyuan/Data/CIFAR_10_data/images/train.txt',
                           transform=transform, frequency='low')


high_prefix = '/home/haojieyuan/RobustFeatures/samples/cifar10/high'
low_prefix = '/home/haojieyuan/RobustFeatures/samples/cifar10/low'

dataloader = torch.utils.data.DataLoader(high_dataset, batch_size=1, shuffle=False)
for img, label, name in dataloader:
    #pdb.set_trace()
    save_img_tensor(img, os.path.join(high_prefix, name[0]))

dataloader = torch.utils.data.DataLoader(low_dataset, batch_size=1, shuffle=False)
for img, label, name in dataloader:
    save_img_tensor(img, os.path.join(low_prefix, name[0]))

