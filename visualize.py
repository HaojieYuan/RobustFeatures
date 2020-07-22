
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from DCT import DCTTransform

import pdb

def save_img_tensor(x, name):
    x_numpy = x.detach().cpu().squeeze(0).permute(1,2,0).numpy()
    x_numpy = x_numpy*255
    im = Image.fromarray(np.uint8(x_numpy))
    im.save(name, 'PNG')




if __name__ == '__main__':
    input_x = './test.png'
    x = Image.open(input_x)
    to_tensor = transforms.Compose([transforms.ToTensor()])
    x = to_tensor(x)
    x = x.unsqueeze(0)


    transform = DCTTransform(channels=3)
    dct_data = transform.dct_batch(x)
    idct_x = transform.idct_batch(dct_data)
    save_img_tensor(idct_x, 'idct.png')

    x_low, x_high = transform.dct_split(x, ratio=0.5, to_rgb=True)
    save_img_tensor(x_low+x_high, 'low+high.png')
    save_img_tensor(x_low, 'low.png')
    save_img_tensor(x_high, 'high.png')

    low = Image.open('low.png')
    high = Image.open('high.png')
    low_numpy = np.array(low)
    high_numpy = np.array(high)
    plus = low_numpy + high_numpy
    im_plus = Image.fromarray(np.uint8(plus))
    im_plus.save('save_and_read_low_high.png', 'PNG')
