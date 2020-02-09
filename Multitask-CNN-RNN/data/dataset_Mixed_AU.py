import os.path
import torchvision.transforms as transforms
from .dataset import DatasetBase
from PIL import Image
import random
import numpy as np
import pickle
import pandas as pd
from PATH import PATH
import torch
PRESET_VARS = PATH()

class dataset_Mixed_AU(DatasetBase):
    def __init__(self, opt, train_mode='Train', transform = None):
        super(dataset_Mixed_AU, self).__init__(opt, train_mode, transform)
        self._name = 'dataset_Mixed_AU'
        self._train_mode = train_mode
        if transform is not None:
            self._transform = transform  
        # read dataset
        self._read_dataset_paths()
    def _get_all_label(self):
        return self._data['label']
    def __getitem__(self, index):
        assert (index < self._dataset_size)
        # start_time = time.time()
        images = []
        labels = []
        img_paths = []
        frames_ids = []
        df = self.sample_seqs[index]
        for i,row in df.iterrows():
            img_path = row['path']
            image = Image.open(img_path).convert('RGB')
            image = self._transform(image)
            label = row[PRESET_VARS.Aff_wild2.categories['AU']].values.astype(np.float32)
            frame_id = row['frames_ids']
            images.append(image)
            labels.append(label)
            img_paths.append(img_path)
            frames_ids.append(frame_id)
        # pack data
        sample = {'image': torch.stack(images,dim=0),
                  'label': np.array(labels),
                  'path': img_paths,
                  'index': index,
                  'id':frames_ids
                  }
        # print (time.time() - start_time)
        return sample
    def _read_dataset_paths(self):
        self._data = self._read_path_label(PRESET_VARS.Aff_wild2.data_file)
        #sample them 
        seq_len = self._opt.seq_len
        self.sample_seqs = []
        if self._train_mode == 'Train':
            N = seq_len//2
        else:
            N = seq_len
        for video in self._data.keys():
            data = self._data[video]
            for i in range(len(data)//N):
                start, end = i*N, i*N + seq_len
                if end >= len(data):
                    start, end = len(data) - seq_len, len(data)
                new_df = data.iloc[start:end]
                if not len(new_df) == seq_len:
                    assert len(new_df) < seq_len
                    count = seq_len - len(new_df)
                    for _ in range(count):
                        new_df = new_df.append(new_df.iloc[-1])
                assert len(new_df) == seq_len
                self.sample_seqs.append(new_df)
        self._ids = np.arange(len(self.sample_seqs)) 
        self._dataset_size = len(self._ids)

    def __len__(self):
    	return self._dataset_size
    def _read_path_label(self, file_path):
        data = pickle.load(open(file_path, 'rb'))
        data = data['AU_Set']
        # read frames ids
        if self._train_mode == 'Train':
            data = data['Training_Set']
        elif self._train_mode == 'Validation':
            data = data['Validation_Set']
        else:
            raise ValueError("train mode must be in : Train, Validation")
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