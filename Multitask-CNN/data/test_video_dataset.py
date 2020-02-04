import os.path
import torchvision.transforms as transforms
from .dataset import DatasetBase
from PIL import Image
import random
import numpy as np
import pickle
import pandas as pd
from PATH import PATH
PRESET_VARS = PATH()
import pickle
class Test_dataset(object):
    def __init__(self, opt, video_data,  train_mode = 'Test', transform = None):
        self._name = 'Test_dataset'
        self._train_mode = train_mode
        if transform is not None:
            self._transform = transform 
        else:
            self._transform = self._create_transform()
        # read dataset
        self._data = video_data
        self._read_dataset()
    def __getitem__(self, index):
        assert (index < self._dataset_size)
        image = None
        label = None
        img_path = self._data['path'][index]
        image = Image.open( img_path).convert('RGB')
        label = self._data['label'][index]
        frame_id = self._data['frames_ids'][index]

        # transform data
        image = self._transform(image)
        # pack data
        sample = {'image': image,
                  'label': label,
                  'path': img_path,
                  'index': index,
                  'frames_ids': frame_id
                  }

        return sample
        
    def _read_dataset(self):        
        self._ids = np.arange(len(self._data['path'])) 
        self._dataset_size = len(self._ids)
        
    def __len__(self):
    	return self._dataset_size
    def _create_transform(self):
        if self._train_mode == 'Train':
            img_size = self._opt.image_size
            resize = int(img_size * 1.2)
            transform_list = [transforms.Resize(resize),
                              transforms.RandomCrop(img_size),
                              transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225]),
                            ]
        else:
            img_size = self._opt.image_size
            transform_list = [transforms.Resize(img_size),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225]),
                            ]
        self._transform = transforms.Compose(transform_list)


