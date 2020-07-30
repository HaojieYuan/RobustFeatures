import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from DCT import DCTTransform
import multiprocessing
from visualize import save_img_tensor
import os
from tqdm import tqdm

import pdb

list_file = '/home/haojieyuan/Data/CIFAR_10_data/images/train.txt'

IN_PREFIX = '/home/haojieyuan/Data/CIFAR_10_data/images/train'
OUT_PREFIX_HIGH = '/home/haojieyuan/Data/CIFAR_10_data/images_high/train'
OUT_PREFIX_LOW  = '/home/haojieyuan/Data/CIFAR_10_data/images_low/train'


thread_num = 5


def split_all_in_list(list_name, partition_num):
    transform3 = DCTTransform(channels=3)
    transform1 = DCTTransform(channels=1)
    f = open(list_name)
    for line in tqdm(f, total=partition_num):
        img_name = line.strip()

        img_in = os.path.join(IN_PREFIX, img_name)
        try:
            x = Image.open(img_in)

            # process images with alpha channel
            #if x.mode == 'RGBA':
            #    x_numpy = np.array(x)
            #    r, g, b, a = np.rollaxis(x_numpy, axis=-1)
            #    x = np.dstack([r, g, b])
            #    x = Image.fromarray(x, 'RGB')

            to_tensor = transforms.Compose([transforms.ToTensor()])
            x = to_tensor(x)
            x = x.unsqueeze(0)

            try:
                x_low, x_high = transform3.dct_split(x, ratio=0.5, to_rgb=True)
            except:
                x_low, x_high = transform1.dct_split(x, ratio=0.5, to_rgb=True)

            img_name = img_name[:-3]+'png'
            img_out_high = os.path.join(OUT_PREFIX_HIGH, img_name)
            img_out_low = os.path.join(OUT_PREFIX_LOW, img_name)
            save_img_tensor(x_low, img_out_low)
            save_img_tensor(x_high, img_out_high)

        except:
            pass

    f.close()

def get_all_lists(list_file, thread_num):
    f = open(list_file)
    i = 0
    for line in f:
        i = i+1
    f.close()

    partition_num = i//thread_num

    out_id = 0
    f = open(list_file)
    i = 0
    out_f = False
    for line in f:
        if i%partition_num == 0:
            if out_f:
                out_f.close()
            out_f = open(list_file+'tmp_{}'.format(out_id),'w')
            out_id = out_id +1
        out_f.write(line)
        i = i+1

    f.close()
    out_f.close()

    return partition_num, out_id-1


def clean_all_lists(list_file, out_id):
    for i in range(out_id+1):
        os.remove(list_file+'tmp_{}'.format(i))

partition_num, out_id = get_all_lists(list_file, thread_num)


threads = []

for i in range(out_id+1):
    threads.append(multiprocessing.Process(target=split_all_in_list, args=(list_file+'tmp_{}'.format(i), partition_num)))

for t in threads:
    t.start()

for t in threads:
    t.join()

clean_all_lists(list_file, out_id)