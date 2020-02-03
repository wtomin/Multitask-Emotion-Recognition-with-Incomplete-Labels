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

class dataset_Mixed_EXPR(DatasetBase):
    def __init__(self, opt, train_mode='Train', transform = None):
        super(dataset_Mixed_EXPR, self).__init__(opt, train_mode, transform)
        self._name = 'dataset_Mixed_EXPR'
        self._train_mode = train_mode
        if transform is not None:
            self._transform = transform  
        # read dataset
        self._read_dataset_paths()
    def __getitem__(self, index):
        assert (index < self._dataset_size)
        # start_time = time.time()
        image = None
        label = None
        img_path = self._data['path'][index]
        image = Image.open(img_path).convert('RGB')
        label = self._data['label'][index]
        image = self._transform(image)
        # pack data
        sample = {'image': image,
                  'label': label,
                  'path': img_path,
                  'index': index 
                  }
        # print (time.time() - start_time)
        return sample
    def _read_dataset_paths(self):
        if not self._train_mode == 'Test':
            self._data = self._read_path_label(PRESET_VARS.Mixed_EXPR.data_file)
        else:
            self._data = self._read_path_label(PRESET_VARS.Aff_wild2.test_data_file)
        self._ids = np.arange(len(self._data['label'])) 
        self._dataset_size = len(self._ids)
    def __len__(self):
    	return self._dataset_size
    def _read_path_label(self, file_path):
        data = pickle.load(open(file_path, 'rb'))
        # read frames ids
        if self._train_mode == 'Train':
            data = data['Training_Set']
        elif self._train_mode == 'Validation':
            data = data['Validation_Set']
        elif self._train_mode == 'Test':
            data = data['Test_Set']
        else:
            raise ValueError("train mode must be in : Train, Validation, Test")
        return data

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
