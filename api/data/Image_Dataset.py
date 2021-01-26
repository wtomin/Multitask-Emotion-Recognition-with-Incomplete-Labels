import pandas as pd
from PIL import Image
import numpy as np
import os
import glob
class Image_Dataset(object):
    def __init__(self, 
        image_dir, 
        transform = None,
        image_ext = ['.jpg', '.bmp', '.png']):
        assert transform is not None
        self._transform = transform
        self.image_dir = image_dir
        self.image_ext = image_ext
        # read dataset
        self._read_dataset()

    def __getitem__(self, index):
        assert (index < self._dataset_size)
        image = None
        label = None
        img_path = self._data['path'][index]
        image = Image.open( img_path).convert('RGB')
        frame_ids = self._data['frames_ids'][index]
        # transform data
        image = self._transform(image)
        # pack data
        sample = {'image': image,
                  'path': img_path,
                  'index': index,
                  'frames_ids': frame_ids
                  }
        return sample
    def _read_dataset(self):
        #sample them 
        frames_paths = glob.glob(os.path.join(self.image_dir, '*'))
        frames_paths = [x for x in frames_paths if any([ext in x for ext in self.image_ext])]
        frames_paths = sorted(frames_paths, key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[-1]))
        self._data = {'path': frames_paths, 'frames_ids': [int(os.path.basename(p).split('.')[0].split('_')[-1]) for p in frames_paths]} # dataframe are easier for indexing
        self._ids = np.arange(len(self._data['path'])) 
        self._dataset_size = len(self._ids) 


    def __len__(self):
        return self._dataset_size
