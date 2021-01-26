import os
import six
import random
import sys
sys.path.append("..")
from config import MODEL_DIR
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from .data_utils import RandomCrop, RandomHorizontalFlip
import torch.nn.functional as F
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
