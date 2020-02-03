import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import os
import os.path


class DatasetFactory:
    def __init__(self):
        pass
    @staticmethod
    def get_by_name(dataset_name, opt, train_mode='Train', transform = None):
        if dataset_name == 'Mixed_EXPR':
            from data.dataset_Mixed_EXPR import dataset_Mixed_EXPR
            dataset = dataset_Mixed_EXPR(opt, train_mode, transform)
        elif dataset_name == 'Mixed_AU':
            from data.dataset_Mixed_AU import dataset_Mixed_AU
            dataset = dataset_Mixed_AU(opt, train_mode, transform)
        elif dataset_name == 'Mixed_VA':
            from data.dataset_Mixed_VA import dataset_Mixed_VA
            dataset = dataset_Mixed_VA(opt, train_mode, transform)
        else:
            raise ValueError("Dataset [%s] not recognized." % dataset_name)

        print('Dataset {} was created'.format(dataset.name))
        return dataset


class DatasetBase(data.Dataset):
    def __init__(self, opt, train_mode='Train', transform=None):
        super(DatasetBase, self).__init__()
        self._name = 'BaseDataset'
        self._root = None
        self._opt = opt
        self._transform = None
        self._train_mode = None
        self._create_transform()

        self._IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG',
            '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
        ]

    @property
    def name(self):
        return self._name

    @property
    def path(self):
        return self._root

    def _create_transform(self):
        self._transform = transforms.Compose([])

    def get_transform(self):
        return self._transform

    def _is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in self._IMG_EXTENSIONS)

    def _is_csv_file(self, filename):
        return filename.endswith('.csv')

    def _get_all_files_in_subfolders(self, dir, is_file):
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

        return images
