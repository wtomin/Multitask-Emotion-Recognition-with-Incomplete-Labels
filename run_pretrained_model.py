import argparse
import os
import torch
import torch.nn as nn
import os.path
from PIL import Image
import random
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import glob
import six
from collections import OrderedDict
import torchvision.transforms as transforms
import numbers
from tqdm import tqdm
import pandas as pd
from scipy.special import softmax
MODEL_DIR = '/media/Samsung/pytorch-benchmarks/models/'
CATEGORIES = {'AU': ['AU1', 'AU2', 'AU4', 'AU6', 'AU12', 'AU15', 'AU20', 'AU25'],
                            'EXPR':['Neutral','Anger','Disgust','Fear','Happiness','Sadness','Surprise'],
                            'VA':['valence', 'arousal']}
Best_AU_Thresholds = {'CNN': [0.1448537, 0.03918985, 0.13766725, 0.02652811, 0.40589422, 0.15572545,0.04808964, 0.10848708],
                      'CNN-RNN': {32: [0.4253935, 0.02641966, 0.1119782, 0.02978198, 0.17256933, 0.06369855, 0.07433069, 0.13828614],
                                  16: [0.30485213, 0.09509478, 0.59577084, 0.4417419, 0.4396544, 0.0452404,0.05204154, 0.0633798 ],
                                  8: [0.4365209 ,0.10177602, 0.2649502,  0.22586018, 0.3772219,  0.07532539, 0.07667687, 0.04306327]}}
################################################## Dataset ############################################
class Image_dataset(object):
    def __init__(self, opt, transform = None):
        self._opt = opt
        assert transform is not None
        self._transform = transform
        # read dataset
        self._read_dataset()

    def __getitem__(self, index):
        assert (index < self._dataset_size)
        if 'RNN' in self._opt.model_type:
            images = []
            labels = []
            img_paths = []
            frames_ids = []
            df = self.sample_seqs[index]
            for i, row in df.iterrows():
                img_path = row['path']
                image = Image.open(img_path).convert('RGB')
                image = self._transform(image)
                frame_id = row['frames_ids']
                images.append(image)
                img_paths.append(img_path)
                frames_ids.append(frame_id)
            # pack data
            sample = {'image': torch.stack(images,dim=0),
                      'path': img_paths,
                      'index': index,
                      'frames_ids':frames_ids
                      }
        else:
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
        seq_len = self._opt.seq_len
        model_type = self._opt.model_type
        frames_paths = glob.glob(os.path.join(self._opt.image_dir, '*'))
        frames_paths = [x for x in frames_paths if any([ext in x for ext in self._opt.image_ext])]
        frames_paths = sorted(frames_paths)
        self._data = {'path': frames_paths, 'frames_ids': np.arange(len(frames_paths))} # dataframe are easier for indexing
        if 'RNN' in self._opt.model_type:
            self._data = pd.DataFrame.from_dict(self._data)
            self.sample_seqs = []
            N = seq_len
            for i in range(len(self._data['path'])//N + 1):
                start, end = i*N, i*N + seq_len
                if end >= len(self._data):
                    start, end = len(self._data) - seq_len, len(self._data)
                new_df = self._data.iloc[start:end]
                if not len(new_df) == seq_len:
                    assert len(new_df) < seq_len
                    count = seq_len - len(new_df)
                    for _ in range(count):
                        new_df = new_df.append(new_df.iloc[-1])
                assert len(new_df) == seq_len
                self.sample_seqs.append(new_df)
            self._ids = np.arange(len(self.sample_seqs)) 
            self._dataset_size = len(self._ids)
        else:
            self._ids = np.arange(len(self._data['path'])) 
            self._dataset_size = len(self._ids) 

    def __len__(self):
        return self._dataset_size
################################################## Model Utils ############################################
def sigmoid(x):
    return 1/(1+np.exp(-x))
def compose_transforms(meta, center_crop=True, new_imageSize = None,
                      override_meta_imsize=False):
    """Compose preprocessing transforms for model

    The imported models use a range of different preprocessing options,
    depending on how they were originally trained. Models trained in MatConvNet
    typically require input images that have been scaled to [0,255], rather
    than the [0,1] range favoured by PyTorch.

    Args:
        meta (dict): model preprocessing requirements
        resize (int) [256]: resize the input image to this size
        center_crop (bool) [True]: whether to center crop the image
        override_meta_imsize (bool) [False]: if true, use the value of `new_meta`
           to select the image input size, 
    Return:
        (transforms.Compose): Composition of preprocessing transforms
    """
    normalize = transforms.Normalize(mean=meta['mean'], std=meta['std'])
    im_size = meta['imageSize']
    if override_meta_imsize:
        im_size = new_imageSize
    assert im_size[0] == im_size[1], 'expected square image size'

    if center_crop:
        transform_list = [transforms.Resize(int(im_size[0]*1.2)),
                          transforms.CenterCrop(size=(im_size[0], im_size[1]))]
    else:
        transform_list = [transforms.Resize(size=(im_size[0], im_size[1]))]
    transform_list += [transforms.ToTensor()]
    if meta['std'] == [1,1,1]: # common amongst mcn models
        transform_list += [lambda x: x * 255.0]
    transform_list.append(normalize)
    return transforms.Compose(transform_list)

def augment_transforms(meta, random_crop=True, new_imageSize = None,
                      override_meta_imsize=False):
    normalize = transforms.Normalize(mean=meta['mean'], std=meta['std'])
    im_size = meta['imageSize']
    if override_meta_imsize:
        im_size = new_imageSize
    assert im_size[0] == im_size[1], 'expected square image size'
    if random_crop:
        v = random.random()
        transform_list = [transforms.Resize(int(im_size[0]*1.2)),
                          RandomCrop(im_size[0], v),
                          RandomHorizontalFlip(v)]
    else:
        transform_list = [transforms.Resize(size=(im_size[0], im_size[1]))]
    transform_list += [transforms.ToTensor()]
    if meta['std'] == [1,1,1]: # common amongst mcn models
        transform_list += [lambda x: x * 255.0]
    transform_list.append(normalize)
    return transforms.Compose(transform_list) 
def load_module_2or3(model_name, model_def_path):
    """Load model definition module in a manner that is compatible with
    both Python2 and Python3

    Args:
        model_name: The name of the model to be loaded
        model_def_path: The filepath of the module containing the definition

    Return:
        The loaded python module."""
    if six.PY3:
        import importlib.util
        spec = importlib.util.spec_from_file_location(model_name, model_def_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        import importlib
        dirname = os.path.dirname(model_def_path)
        sys.path.insert(0, dirname)
        module_name = os.path.splitext(os.path.basename(model_def_path))[0]
        mod = importlib.import_module(module_name)
    return mod
def load_model(model_name, MODEL_DIR):
    """Load imoprted PyTorch model by name

    Args:
        model_name (str): the name of the model to be loaded

    Return:
        nn.Module: the loaded network
    """
    model_def_path = os.path.join(MODEL_DIR, model_name + '.py')
    weights_path = os.path.join(MODEL_DIR, model_name + '.pth')
    mod = load_module_2or3(model_name, model_def_path)
    func = getattr(mod, model_name)
    net = func(weights_path=weights_path)
    return net
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x
################################################Data Utils ######################################################
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


################################################## Model: ResNet50 ############################################
class Head(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_class = 8):
        super(Head, self).__init__()
        self._name = 'Head'
        self.bn0 = nn.BatchNorm1d(input_dim)
        self.fc_0 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc_1 = nn.Linear(hidden_dim, n_class)
    def forward(self, x):
        x = self.bn0(x)
        f0 = self.bn1(F.relu(self.fc_0(x)))
        output = self.fc_1(f0)
        return {'output':output, 'feature':f0}
class BackBone(nn.Module):
    def __init__(self, opt):
        super(BackBone, self).__init__()
        self._name = 'BackBone'
        self._opt = opt
        self.model = self._init_create_networks()

    def _init_create_networks(self):
        # the feature extractor
        # different models have different input sizes, different mean and std
        if self._opt.pretrained_dataset == 'ferplus' or self._opt.pretrained_dataset == 'sfew':
            if self._opt.pretrained_dataset == 'ferplus':
                model_name = 'resnet50_ferplus_dag'
                model_dir = os.path.join(MODEL_DIR, 'fer+')
            else:
                model_name = 'resnet50_face_sfew_dag'
                model_dir = os.path.join(MODEL_DIR, 'sfew')
            feature_extractor = load_model(model_name, model_dir)
            meta = feature_extractor.meta
            if not meta['imageSize'][0] == self._opt.image_size:
                new_imageSize = [self._opt.image_size, self._opt.image_size, 3]
                override_meta_imsize = True
            else:
                new_imageSize = None
                override_meta_imsize = False
            setattr(self, 'augment_transforms', augment_transforms(meta, new_imageSize=new_imageSize, override_meta_imsize=override_meta_imsize))
            setattr(self, 'compose_transforms', compose_transforms(meta, new_imageSize=new_imageSize, override_meta_imsize=override_meta_imsize))
        elif self._opt.pretrained_dataset == 'imagenet':
            import torchvision.models as models
            model_name = 'resnet50_imagenet'
            feature_extractor = models.resnext50_32x4d(pretrained=True)
            im_size = self._opt.image_size
            transform_list = transforms.Compose([
                            transforms.Resize(int(im_size*1.2)),
                            transforms.CenterCrop(im_size),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])
            setattr(self, 'compose_transforms', transform_list)
            transform_list = transforms.Compose([
                            transforms.Resize(int(im_size*1.2)),
                            transforms.RandomCrop(im_size),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])
            setattr(self, 'augment_transforms', transform_list)
        else:
            raise ValueError("Pretrained dataset %s not recognized." % pretrained_dataset)
        setattr(feature_extractor, 'name', model_name)
        # reform the final layer of feature extrator, turn it into a Identity module
        last_layer_name, last_module = list(feature_extractor.named_modules())[-1]
        try:
            in_channels, out_channels = last_module.in_features, last_module.out_features
            last_linear = True
        except:
            in_channels, out_channels = last_module.in_channels, last_module.out_channels
            last_linear = False
        setattr(feature_extractor, '{}'.format(last_layer_name), Identity()) # the second last layer has 512 dimensions
        setattr(self, 'output_feature_dim', in_channels)

        # orginal input size is 224, if the image size is different from 224, change the pool5 layer to adaptive avgpool2d
        if not meta['imageSize'][0] == self._opt.image_size:
            pool_layer_name, pool_layer = list(feature_extractor.named_modules())[-2]
            setattr(feature_extractor, '{}'.format(pool_layer_name), nn.AdaptiveAvgPool2d((1, 1)))
        return feature_extractor

    def forward(self, x):
        return self.model(x) 

class Model(nn.Module):
    def __init__(self, backbone, classifier, sofar_task):
        super(Model, self).__init__()
        self._name = 'Model'
        self.backbone = backbone
        self.classifier = classifier
        self.sofar_task = sofar_task
    def forward(self, x):
        f = self.backbone(x).squeeze(-1).squeeze(-1)
        features = {'cross_task': f}
        outputs = {}
        for i,m in enumerate(self.classifier):
            task = self.sofar_task[i] 
            o = m(f)
            outputs[task] = o['output']
            features[task] = o['feature']
        return {'output':outputs, 'feature':features}
class ResNet50():
    def __init__(self, opt):
        self._opt = opt
        self._name = 'ResNet50'
        self._output_size_per_task = {'AU': self._opt.AU_label_size, 'EXPR': self._opt.EXPR_label_size, 'VA': self._opt.VA_label_size * self._opt.digitize_num}
        # create networks
        self._init_create_networks()

    def _init_create_networks(self):
        """
        init current model according to sofar tasks
        """
        backbone = BackBone(self._opt)
        output_sizes = [self._output_size_per_task[x] for x in self._opt.tasks]
        output_feature_dim = backbone.output_feature_dim
        classifiers = [Head(output_feature_dim, self._opt.hidden_size, output_sizes[i]) for i in range(len(self._opt.tasks))]
        classifiers = nn.ModuleList(classifiers)
        self.resnet50 = Model(backbone, classifiers, self._opt.tasks)
        if len(self._opt.gpu_ids) > 1:
            self.resnet50 = torch.nn.DataParallel(self.resnet50, device_ids=self._opt.gpu_ids)
        self.resnet50.cuda()
    def load(self, model_path):
        self.resnet50.load_state_dict(torch.load(model_path))  

    def set_eval(self):
        self.resnet50.eval()
        self._is_train = False

    def forward(self, input_image = None):
        assert self._is_train is False, "Model must be in eval mode"
        with torch.no_grad():
            input_image = Variable(input_image)
            if not input_image.is_cuda:
                input_image = input_image.cuda()
            output = self.resnet50(input_image)
            out_dict = self._format_estimates(output['output'])
            out_dict_raw = dict([(key,output['output'][key].cpu().numpy()) for key in output['output'].keys()])
        return out_dict, out_dict_raw
    def _format_estimates(self, output):
        estimates = {}
        for task in output.keys():
            if task == 'AU':
                o = (torch.sigmoid(output['AU'].cpu())>0.5).type(torch.LongTensor)
                estimates['AU'] = o.numpy()
            elif task == 'EXPR':
                o = F.softmax(output['EXPR'].cpu(), dim=-1).argmax(-1).type(torch.LongTensor)
                estimates['EXPR'] = o.numpy()
            elif task == 'VA':
                N = self._opt.digitize_num
                v = F.softmax(output['VA'][:, :N].cpu(), dim=-1).numpy()
                a = F.softmax(output['VA'][:, N:].cpu(), dim=-1).numpy()
                bins = np.linspace(-1, 1, num=self._opt.digitize_num)
                v = (bins * v).sum(-1)
                a = (bins * a).sum(-1)
                estimates['VA'] = np.stack([v, a], axis = 1)
        return estimates
############################################ CNN-RNN ########################################33
class GRU_Head(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_class = 8):
        super(GRU_Head, self).__init__()
        self._name = 'Head'
        self.GRU_layer = nn.GRU(input_dim, hidden_dim, batch_first= True, bidirectional=True)
        self.fc_1 = nn.Linear(hidden_dim*2, n_class)
    def forward(self, x):
        B, N, C = x.size()
        self.GRU_layer.flatten_parameters()
        f0 = F.relu(self.GRU_layer(x)[0])
        output = self.fc_1(f0)
        return {'output':output, 'feature':f0}

class Seq_Model(nn.Module):
    def __init__(self, backbone, classifier, sofar_task):
        super(Seq_Model, self).__init__()
        self._name = 'Seq_Model'
        self.backbone = backbone
        self.classifier = classifier
        self.sofar_task = sofar_task
    def forward(self, x):
        B, N, C, W, H = x.size()
        x = x.view(B*N, C, W, H)
        out_backbone = self.backbone(x)
        outputs = {}
        features = {}
        for i,m in enumerate(self.classifier):
            task = self.sofar_task[i] 
            feature = out_backbone['feature'][task]
            feature = feature.view(B, N ,-1)
            o = m(feature)
            outputs[task] = o['output']
            features[task] = feature
        return {'output':outputs, 'feature':features}

class ResNet50_GRU():
    def __init__(self, opt):
        self._opt = opt
        self._name = 'ResNet50_GRU'
        self._output_size_per_task = {'AU': self._opt.AU_label_size, 'EXPR': self._opt.EXPR_label_size, 'VA': self._opt.VA_label_size * self._opt.digitize_num}
        # create networks
        self._init_create_networks()
    def _init_create_networks(self):
        """
        init current model according to sofar tasks
        """
        backbone = BackBone(self._opt)
        output_sizes = [self._output_size_per_task[x] for x in self._opt.tasks]
        output_feature_dim = backbone.output_feature_dim
        classifiers = [Head(output_feature_dim, self._opt.hidden_size, output_sizes[i]) for i in range(len(self._opt.tasks))]
        classifiers = nn.ModuleList(classifiers)
        resnet50 = Model(backbone, classifiers, self._opt.tasks)
        # create GRUs 
        GRU_classifiers = [GRU_Head(self._opt.hidden_size, self._opt.hidden_size//2, output_sizes[i]) for i in range(len(self._opt.tasks))]
        GRU_classifiers = nn.ModuleList(GRU_classifiers)
        self.resnet50_GRU = Seq_Model(resnet50, GRU_classifiers, self._opt.tasks)
        if len(self._opt.gpu_ids) > 1:
            self.resnet50_GRU = torch.nn.DataParallel(self.resnet50_GRU, device_ids=self._opt.gpu_ids)
        self.resnet50_GRU.cuda()
    def load(self, model_path):
        self.resnet50_GRU.load_state_dict(torch.load(model_path))  
    def set_eval(self):
        self.resnet50_GRU.eval()
        self._is_train = False

    def forward(self, input_image = None):
        assert self._is_train is False, "Model must be in eval mode"
        with torch.no_grad():
            input_image = Variable(input_image)
            if not input_image.is_cuda:
                input_image = input_image.cuda()
            output = self.resnet50_GRU(input_image)
            out_dict = self._format_estimates(output['output'])
            out_dict_raw = dict([(key, output['output'][key].cpu().numpy()) for key in output['output'].keys()])
        return out_dict, out_dict_raw
    def _format_estimates(self, output):
        estimates = {}
        for task in output.keys():
            if task == 'AU':
                o = (torch.sigmoid(output['AU'].cpu())>0.5).type(torch.LongTensor)
                estimates['AU'] = o.numpy()
            elif task == 'EXPR':
                o = F.softmax(output['EXPR'].cpu(), dim=-1).argmax(-1).type(torch.LongTensor)
                estimates['EXPR'] = o.numpy()
            elif task == 'VA':
                N = self._opt.digitize_num
                v = F.softmax(output['VA'][:,:, :N].cpu(), dim=-1).numpy()
                a = F.softmax(output['VA'][:,:, N:].cpu(), dim=-1).numpy()
                bins = np.linspace(-1, 1, num=self._opt.digitize_num)
                v = (bins * v).sum(-1)
                a = (bins * a).sum(-1)
                estimates['VA'] = np.stack([v, a], axis = -1)
        return estimates
parser = argparse.ArgumentParser(description='Implement the pretrained model on your own data')
parser.add_argument('--image_dir', type=str, 
                    help='a directory containing a sequence of cropped and aligned face images')
parser.add_argument('--model_type', type=str, default='CNN', choices= ['CNN', 'CNN-RNN'],
                    help='By default, the CNN pretrained models are stored in "Multitask-CNN", and the CNN-RNN \
                    pretrained models are stored in "Multitask-CNN-RNN"')
parser.add_argument('--seq_len', type=int, default = 32, choices = [32, 16, 8], help='sequence length when the model type is CNN-RNN')
parser.add_argument('--image_ext', default = ['.jpg', '.bmp', '.png'], help='image extentions')
parser.add_argument('--eval_with_teacher', action='store_true', help='whether to predict with teacher model')
parser.add_argument('--eval_with_students', action='store_true', help='whether to predict with student models')
parser.add_argument('--ensemble', action='store_true', help='whether to merge the student predictions')
parser.add_argument('--AU_label_size', type=int, default = 8, help='# of AUs')
parser.add_argument('--EXPR_label_size', type=int, default = 7, help='# of EXpressions')
parser.add_argument('--VA_label_size', type=int, default = 2, help='# of VA ')
parser.add_argument('--digitize_num', type=int, default= 20, choices = [1, 20], help='1 means no digitization,\
                                                 20 means to digitize continuous label to 20 one hot vector ')
parser.add_argument('--hidden_size', type=int, default = 128, help='the embedding size of each output head' )
parser.add_argument('--image_size', type=int, default= 112, help='input image size')
parser.add_argument('--batch_size', type=int, default= 20, help='input batch size per task')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--workers', type=int, default=0, help='number of workers')
parser.add_argument('--tasks', type=str, default = ['EXPR','AU','VA'],nargs="+")
parser.add_argument('--save_dir', type=str, help='where to save the predictions')
parser.add_argument('--pretrained_dataset', type=str, default='ferplus',
                                  choices = ['ferplus', 'sfew','imagenet'], 
                                  help="the pretrained_dataset of the face feature extractor, choices:['ferplus', 'sfew','imagenet']")
opt = parser.parse_args()

def test_one_video( model, data_loader):
    track_val = {}
    for task in opt.tasks:
        track_val[task] = {'outputs':[], 'estimates':[], 'frames_ids':[]}
    for i_val_batch, val_batch in tqdm(enumerate(data_loader), total = len(data_loader)):
        estimates, outputs = model.forward( input_image = val_batch['image'])
        #store the predictions and labels
        for task in opt.tasks:
            if 'RNN' in opt.model_type:
                B, N, C = outputs[task].shape
                track_val[task]['outputs'].append(outputs[task].reshape(B*N, C))
                track_val[task]['frames_ids'].append(np.array([np.array(x) for x in val_batch['frames_ids']]).reshape(B*N, -1).squeeze())
                track_val[task]['estimates'].append(estimates[task].reshape(B*N, -1).squeeze())
            else:
                track_val[task]['outputs'].append(outputs[task])
                track_val[task]['frames_ids'].append(np.array(val_batch['frames_ids']))
                track_val[task]['estimates'].append(estimates[task])
        # if i_val_batch >5:
        #     break
    for task in opt.tasks:
        for key in track_val[task].keys():
            track_val[task][key] = np.concatenate(track_val[task][key], axis=0)
    #assert len(track_val['frames_ids']) -1 == track_val['frames_ids'][-1]
    return track_val
def save_to_file(frames_ids, predictions, save_path, task= 'AU'):
    save_dir = os.path.dirname(os.path.abspath(save_path))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    categories = CATEGORIES[task]
    #filtered out repeated frames
    mask = np.zeros_like(frames_ids, dtype=bool)
    mask[np.unique(frames_ids, return_index=True)[1]] = True
    frames_ids = frames_ids[mask]
    predictions = predictions[mask]
    assert len(frames_ids) == len(predictions)
    with open(save_path, 'w') as f:
        f.write(",".join(categories)+"\n")
        for i, line in enumerate(predictions):
            if isinstance(line, np.ndarray):
                digits = []
                for x in line:
                    if isinstance(x, float) or isinstance(x, np.float32) or isinstance(x, np.float64):
                        digits.append("{:.4f}".format(x))
                    elif isinstance(x, np.int64):
                        digits.append(str(x))
                line = ','.join(digits)+'\n'
            elif isinstance(line, np.int64):
                line = str(line)+'\n'
            if i == len(predictions)-1:
                line = line[:-1]
            f.write(line)    
def main():
    if opt.model_type == 'CNN':
        model = ResNet50(opt)
        val_transforms = model.resnet50.backbone.compose_transforms
    elif opt.model_type == 'CNN-RNN':
        model = ResNet50_GRU(opt)
        val_transforms = model.resnet50_GRU.backbone.backbone.compose_transforms
    model.set_eval()
    model_paths = OrderedDict()
    if opt.eval_with_teacher:
        path = glob.glob(os.path.join('Multitask-'+opt.model_type, '*teacher*'))
        if len(path) == 0:
            path = glob.glob(os.path.join('Multitask-'+opt.model_type, 'Seq_len={}'.format(opt.seq_len), '*teacher*'))
        assert len(path) == 1
        model_paths.update({'teacher': path[0]})
    if opt.eval_with_students:
        paths = sorted(glob.glob(os.path.join('Multitask-'+opt.model_type, '*student*')))
        if len(paths) == 0:
            paths = glob.glob(os.path.join('Multitask-'+opt.model_type, 'Seq_len={}'.format(opt.seq_len), '*student*'))
        assert len(paths) != 0
        for i in range(len(paths)):
            model_paths.update({'student_{}'.format(i): paths[i]})
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)
    outputs_record = {}
    estimates_record = {}
    frames_ids_record = {}
    for model_id, model_path in model_paths.items():
        model.load(model_path)
        dataset =  Image_dataset(opt, transform=val_transforms)
        dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle= False,
        num_workers=opt.workers,
        drop_last=False)
        outputs_record[model_id] = {}
        estimates_record[model_id] = {}
        frames_ids_record[model_id] = {}
        track = test_one_video(model, dataloader)
        torch.cuda.empty_cache() 
        for task in opt.tasks:
            outputs_record[model_id][task] = track[task]['outputs']
            estimates_record[model_id][task] = track[task]['estimates']
            frames_ids_record[model_id][task] = track[task]['frames_ids']
            save_path = '{}/{}/{}.txt'.format(opt.save_dir, model_id, task)
            save_to_file(track[task]['frames_ids'], track[task]['estimates'], save_path, task=task)
    #merge the raw outputs && save them with raw_outputs
    if opt.ensemble and opt.eval_with_students:
        for task in opt.tasks:
            preds = []
            for model_id in outputs_record.keys():
                if 'student' in model_id:
                    preds.append(outputs_record[model_id][task])
            preds = np.array(preds)
            #assert frames_ids_record[0][task][video] == frames_ids_record[1][task][video]
            video_frames_ids = frames_ids_record[model_id][task]
            if task == 'AU':
                merged_preds = sigmoid(preds)
                merged_preds = np.mean(merged_preds, axis=0)
                save_path = '{}/{}/{}.txt'.format(opt.save_dir, 'merged_raw', task)
                save_to_file(video_frames_ids, merged_preds, save_path, task='AU')
                best_thresholds_over_models = Best_AU_Thresholds[opt.model_type]
                if 'RNN' in opt.model_type:
                    best_thresholds_over_models = best_thresholds_over_models[opt.seq_len]
                #print("The best AU thresholds over models: {}".format(best_thresholds_over_models))
                merged_preds = merged_preds > (np.ones_like(merged_preds)*best_thresholds_over_models)
                merged_preds = merged_preds.astype(np.int64)
                save_path = '{}/{}/{}.txt'.format(opt.save_dir, 'merged', task)
                save_to_file(video_frames_ids, merged_preds, save_path, task='AU')
            elif task == 'EXPR':
                merged_preds = softmax(preds, axis=-1).mean(0)
                save_path = '{}/{}/{}.txt'.format(opt.save_dir, 'merged_raw', task)
                save_to_file(video_frames_ids, merged_preds, save_path, task='EXPR')
                merged_preds = merged_preds.argmax(-1).astype(np.int).squeeze()
                save_path = '{}/{}/{}.txt'.format(opt.save_dir, 'merged', task)
                save_to_file(video_frames_ids, merged_preds, save_path, task='EXPR')
            else:
                N = opt.digitize_num
                v = softmax(preds[:, :, :N], axis=-1)
                a = softmax(preds[:, :, N:], axis=-1)
                bins = np.linspace(-1, 1, num=opt.digitize_num)
                v = (bins * v).sum(-1)
                a = (bins * a).sum(-1)
                merged_preds = np.stack([v.mean(0), a.mean(0)], axis = 1).squeeze() 
                save_path = '{}/{}/{}.txt'.format(opt.save_dir, 'merged', task)
                save_to_file(video_frames_ids, merged_preds, save_path, task='VA') 
                save_path = '{}/{}/{}.txt'.format(opt.save_dir, 'merged_raw', task)
                save_to_file(video_frames_ids, merged_preds, save_path, task='VA') 


if __name__ == '__main__':
    main()


