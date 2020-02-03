from PIL import Image
import numbers 
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import six
import sys
import os
from os.path import join as pjoin
import numpy as np
import random
from PIL import Image
from torchvision.transforms import functional as TF
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
def imshow_grid(images, shape=[2, 2], name='default', save=False):
    """Plot images in a grid of a given shape."""
    fig = plt.figure()
    grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.05)

    size = shape[0] * shape[1]
    for i in range(size):
        grid[i].axis('off')
        img = images[i]
        if img.shape[0]==3:
            img  = img.transpose(1, 2, 0)
        img = (img - img.min())/(img.max() - img.min())
        grid[i].imshow(img, vmin=-132, vmax = 164)  # The AxesGrid object work as a list of axes.

    plt.show() 

class RandomCrop(object):
    def __init__(self, size, v):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.v = v
    def __call__(self, img):

        w, h = img.size
        th, tw = self.size
        x1 = int(( w - tw)*self.v)
        y1 = int(( h - th)*self.v)
        #print("print x, y:", x1, y1)
        assert(img.size[0] == w and img.size[1] == h)
        if w == tw and h == th:
            out_image = img
        else:
            out_image = img.crop((x1, y1, x1 + tw, y1 + th)) #same cropping method for all images in the same group
        return out_image

class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __init__(self, v):
        self.v = v
        return
    def __call__(self, img):
        if self.v < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT) 
        #print ("horiontal flip: ",self.v)
        return img


def split_pil_image_from_array(img_array):
    pil_image_rgb = Image.fromarray(img_array[:, :, :3].astype(np.uint8))
    num_channels = img_array.shape[2]
    pil_images_grey = []
    for i in range(num_channels-3):
        array = img_array[:, :, i+3]
        grey_image = Image.fromarray(array.astype(np.uint8))
        pil_images_grey.append(grey_image)
    return pil_image_rgb, pil_images_grey
class CustomCenterCrop(object):
    def __init__(self, size):
        """
        if size is an int, it is the smallest edge in (heigh, width)
        elif size is a list, which specifizes the height and width
        """
        self.size = size
    def __call__(self, img_array):
        pil_image_rgb, pil_images_grey = split_pil_image_from_array(img_array)
        w, h = pil_image_rgb.size
        if isinstance(self.size, list): 
            th, tw = self.size
        elif isinstance(self.size, numbers.Number):
            if h>=w:
                th, tw = h/float(w)*self.size, self.size
            else:
                th, tw = self.size, w/float(h)*self.size
        else:
            raise ValueError("size should be either int or list of int numbers.")

        x1 = (w - tw) // 2
        y1 = (h - th) // 2
        if not (w == tw and h == th):
            pil_image_rgb = pil_image_rgb.crop((x1, y1, x1 + tw, y1 + th))
        out_images_grey = []
        for grey_img in pil_images_grey:
            assert(grey_img.size[0] == w and grey_img.size[1] == h)
            if w == tw and h == th:
                out_images_grey.append(grey_img)
            else:
                out_images_grey.append(grey_img.crop((x1, y1, x1 + tw, y1 + th)))
        if len(out_images_grey)!=0:
            out_images_grey = np.stack([np.array(x) for x in out_images_grey], axis=-1)
        return np.concatenate((np.array(pil_image_rgb), out_images_grey), axis=-1)
class CustomRandomCrop(object):
    def __init__(self, size, seed=None):
        """
        if size is an int, it is the smallest edge in (heigh, width)
        elif size is a list, which specifizes the height and width
        """
        self.size = size
        self.seed = seed
    def __call__(self, img_array):
        pil_image_rgb, pil_images_grey = split_pil_image_from_array(img_array)
        w, h = pil_image_rgb.size
        if isinstance(self.size, list): 
            th, tw = self.size
        elif isinstance(self.size, numbers.Number):
            if h>=w:
                th, tw = h/w*self.size, self.size
            else:
                th, tw = self.size, w/h*self.size
        else:
            raise ValueError("size should be either int or list of int numbers.")
        if self.seed is not None:
            random.seed(self.seed)
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        if not (w == tw and h == th):
            pil_image_rgb = pil_image_rgb.crop((x1, y1, x1 + tw, y1 + th))
        out_images_grey = []
        for grey_img in pil_images_grey:
            assert(grey_img.size[0] == w and grey_img.size[1] == h)
            if w == tw and h == th:
                out_images_grey.append(grey_img)
            else:
                out_images_grey.append(grey_img.crop((x1, y1, x1 + tw, y1 + th)))
        if len(out_images_grey)!=0:
            out_images_grey = np.stack([np.array(x) for x in out_images_grey], axis=-1)
        return np.concatenate((np.array(pil_image_rgb), out_images_grey), axis=-1)

class CustomRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __init__(self, seed=None):
        self.seed = seed

    def __call__(self, img_array):
        if self.seed is not None:
            random.seed(self.seed)
        v = random.random()
        pil_image_rgb, pil_images_grey = split_pil_image_from_array(img_array)
        if v < 0.5:
            pil_image_rgb = pil_image_rgb.transpose(Image.FLIP_LEFT_RIGHT)
            pil_images_grey = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in pil_images_grey]
            pil_images_grey = np.stack([np.array(x) for x in pil_images_grey], axis=-1)
            return np.concatenate((np.array(pil_image_rgb), pil_images_grey), axis=-1)
        else:
            return img_array
class CustomScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the largest edge if 'size' is an int.
    For example, if height > width, then image will be
    rescaled to (size, size * width/height )
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
    def _resize(self, img):
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and h == self.size) or (h <= w and w == self.size):
                return img
            if w < h:
                oh = self.size
                ow = int(w/h * self.size)
                return img.resize((ow, oh), self.interpolation)
            else:
                ow = self.size
                oh = int(h/w * self.size)
                return img.resize((ow, oh), self.interpolation)
        elif isinstance(self.size, list):
            return img.resize(self.size[::-1], self.interpolation)

    def __call__(self, img_array):
        # the first 3 channels in image array is the rgb image, the rest of channels are heatmaps
        pil_image_rgb_o, pil_images_grey_o = split_pil_image_from_array(img_array)
        pil_image_rgb = self._resize(pil_image_rgb_o)
        pil_images_grey = []
        for grey_image in pil_images_grey_o:
            grey_image = self._resize(grey_image)
            pil_images_grey.append(grey_image)
        if len(pil_images_grey)!=0:
            assert pil_images_grey[0].size == pil_image_rgb.size
            pil_images_grey = np.stack([np.array(x) for x in pil_images_grey], axis=-1)
        return np.concatenate((np.array(pil_image_rgb), pil_images_grey), axis=-1)
class CustomPad(object):
    def __init__(self, size, pad_value = 0):
        """pad input image to
        if size is an int: (size, size, _)
        if size is list with two integers, assert size[0] == size[1] , specifize (h, w)
        """
        self.size = size
        self.pad_value = pad_value
    def __call__(self, img_array):
        pil_image_rgb_o, pil_images_grey_o = split_pil_image_from_array(img_array)
        if isinstance(self.size, numbers.Number):
            size = [self.size, self.size]
        elif isinstance(self.size, list):
            assert len(self.size) == 2
            size = self.size
        w, h = pil_image_rgb_o.size
        assert size[1]>=w and size[0]>=h
        th, tw = (size[0] - h)//2, (size[1] - w)//2
        pil_image_rgb = TF.pad(pil_image_rgb_o, padding=(tw, th, tw+(size[1]- w)%2, th+(size[0] - h)%2))
        assert pil_image_rgb.size[0] == self.size
        pil_images_grey = []
        for grey_image in pil_images_grey_o:
            grey_image = TF.pad(grey_image, padding=(tw, th, tw+(size[1]- w)%2, th+(size[0] - h)%2))
            pil_images_grey.append(grey_image)
        if len(pil_images_grey)!=0:
            assert pil_images_grey[0].size == pil_image_rgb.size
            pil_images_grey = np.stack([np.array(x) for x in pil_images_grey], axis=-1)
        return np.concatenate((np.array(pil_image_rgb), pil_images_grey), axis=-1)
class CustomNormalize(object):
    def __init__(self, mean, std):
        # only do normalization for the RGB channels (the first three channels)
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (3//len(self.mean))
        rep_std = self.std * (3//len(self.std))

        # TODO: make efficient
        for i in range(3):
            tensor[i, :, :].sub_(rep_mean[i]).div_(rep_std[i])
        return tensor
class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, img_array):
        if isinstance(img_array, np.ndarray):
            # handle numpy array
            assert len(img_array.shape) == 3 # H, W, C
            img = torch.from_numpy(img_array).permute(2, 0, 1).contiguous() #channels, W, H
        else:
            raise ValueError("input image has to be ndarray!")
        return img.float().div(255) if self.div else img.float()
