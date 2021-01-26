import torch
import os
from copy import deepcopy
from .Multitask_CNN import ResNet50
from .Multitask_CNN_RNN import ResNet50_GRU
import sys
sys.path.append("..")
from config import OPT, PRETRAINED_MODEL_DIR


class ModelFactory:
    def __init__(self):  
        pass 
    @classmethod
    def get(
        self,
        device,
        Mtype = 'CNN',
        num_models = 1,
        ):
        assert num_models>=1, "At least 1 model in the ensemble."
        if Mtype == 'CNN':
            single_model = ResNet50(device)
        elif Mtype=='CNN_RNN':
            single_model = ResNet50_GRU(device)
        else:
            raise ValueError("{} not supported".format(Mtype))
        model_paths = [os.path.join(PRETRAINED_MODEL_DIR, Mtype, '{}.pth'.format(i)) for i in range(5)]
        model_paths = model_paths[:num_models]
        ensemble = []
        val_transforms = None
        for model_path in model_paths:
            model = deepcopy(single_model)
            model.load(model_path)
            model.set_eval()
            ensemble.append(model)
        if Mtype == 'CNN':
            val_transforms = model.resnet50.backbone.compose_transforms
        else:
            val_transforms = model.resnet50_GRU.backbone.backbone.compose_transforms
        return ensemble, val_transforms



