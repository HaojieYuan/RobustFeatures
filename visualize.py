
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from DCT import DCTTransform
import os

import pdb

def save_img_tensor(x, name):
    if x.shape[1] == 3:
        x_numpy = x.detach().cpu().squeeze(0).permute(1,2,0).numpy()
    elif x.shape[1] == 1:
        x_numpy = x.detach().cpu().squeeze(0).squeeze(0).numpy()
    else:
        assert False, "Unkown image shape: {}".format(x.shape)
    x_numpy = x.detach().cpu().squeeze(0).permute(1,2,0).numpy()
    x_numpy = x_numpy*255
    im = Image.fromarray(np.uint8(x_numpy))
    if not os.path.exists(os.path.dirname(name)):
        try:
            os.makedirs(os.path.dirname(name))
        except:
            pass
    im.save(name, 'PNG')




if __name__ == '__main__':
    input_x = './samples/test.png'
    x = Image.open(input_x)
    to_tensor = transforms.Compose([transforms.ToTensor()])
    x = to_tensor(x)
    x = x.unsqueeze(0)

    pdb.set_trace()


    transform = DCTTransform(channels=3)
    dct_data = transform.dct_batch(x)
    idct_x = transform.idct_batch(dct_data)
    save_img_tensor(idct_x, './samples/idct.png')


    x_low, x_high = transform.dct_split(x, ratio=0.5, to_rgb=True)
    save_img_tensor(x_low+x_high, './samples/low+high.png')
    save_img_tensor(x_low, './samples/low.png')
    save_img_tensor(x_high, './samples/high.png')

    low = Image.open('./samples/low.png')
    high = Image.open('./samples/high.png')
    low_numpy = np.array(low)
    high_numpy = np.array(high)
    plus = low_numpy + high_numpy
    im_plus = Image.fromarray(np.uint8(plus))
    im_plus.save('./samples/save_and_read_low_high.png', 'PNG')
