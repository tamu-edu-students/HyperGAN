"""This module contains simple helper functions """
from __future__ import print_function
import numpy as np
from PIL import Image
import os
import torch
import random
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch
# from visdom import Visdom
import torchvision.transforms as transforms
from skimage.filters import threshold_otsu
from .constants import HyperConstants


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)



to_pil = transforms.ToPILImage()
rgb_to_gray = transforms.Grayscale(num_output_channels=1)

class QueueMask():
    def __init__(self, length):
        self.max_length = length
        self.queue = []

    def insert(self, mask):
        if self.queue.__len__() >= self.max_length:
            self.queue.pop(0)

        self.queue.append(mask)

    def rand_item(self):
        assert self.queue.__len__() > 0, 'Error! Empty queue!'
        return self.queue[np.random.randint(0, self.queue.__len__())]

    def last_item(self):
        assert self.queue.__len__() > 0, 'Error! Empty queue!'
        return self.queue[self.queue.__len__()-1]



def mask_generator(shadow, shadow_free, isHyper=False):
    
    if isHyper:

        im_f = hyper_to_gray(shadow_free)
        im_s = hyper_to_gray(shadow)
    else:
        im_f = rgb_to_gray(mod_to_pil(shadow_free, isHyper))
        im_s = rgb_to_gray(mod_to_pil(shadow, isHyper))

    diff = (np.asarray(im_f, dtype='float32')- np.asarray(im_s, dtype='float32')) # difference between shadow image and shadow_free image
    L = threshold_otsu(diff)
    mask = torch.tensor((np.float32(diff >= L)-0.5)/0.5).unsqueeze(0).unsqueeze(0).cuda() #-1.0:non-shadow, 1.0:shadow
    mask.requires_grad = False

    return mask

def mod_to_pil(tensor, isHyper=False):

    if not isHyper:
        img = 0.5 * (tensor.detach().data + 1.0)
        return (to_pil(img.data.squeeze(0).cpu()))
    else:
        img = 0.5 * (tensor.detach().data + 1.0)
        tensor_permuted = img.data.squeeze(0).cpu().permute(1,2,0)
        arr = tensor_permuted.numpy()

        red = normalize_band(arr[:, :, HyperConstants.RED_BAND])
        green = normalize_band(arr[:, :, HyperConstants.GREEN_BAND])
        blue = normalize_band(arr[:, :, HyperConstants.BLUE_BAND])
        rgb_image = np.dstack((red, green, blue))

        return (to_pil(rgb_image))

def tensor2spectral(tensor, x, y):
    
    arr_list = []

    img = 0.5 * (tensor.detach().data + 1.0)
    tensor_permuted = img.data.squeeze(0).cpu().permute(1,2,0)
    arr = tensor_permuted.numpy()

    for i in range(arr.shape[2]):
        twoD = normalize_band(arr[:, :, i])
        arr_list.append(twoD)
    
    image = np.dstack(arr_list)
    #print(image.shape)
    return image[x, y, :]

def tensor2image(tensor):
    image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)

def normalize_band(band):
    return ((band - band.min()) / (band.max() - band.min()) * 255).astype(np.uint8)

def hyper_to_gray(tensor):
    img = 0.5 * (tensor.detach().data + 1.0)
    tensor_permuted = img.data.squeeze(0).cpu().permute(1,2,0)
    arr = tensor_permuted.numpy()

    return np.mean(arr, axis=2)

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)
    
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)