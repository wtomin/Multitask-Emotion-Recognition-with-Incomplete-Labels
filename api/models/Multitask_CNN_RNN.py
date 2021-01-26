import torch.nn as nn
import torch
import sys
sys.path.append("..")
from config import OPT
from config import MODEL_DIR
from utils import Head, GRU_Head, Seq_Model, BackBone,Model
from torch.autograd import Variable
import torch.nn.functional as F
import copy
import numpy as np
############################################ CNN-RNN ########################################33

class ResNet50_GRU():
    def __init__(self, device):
        self._opt = copy.copy( OPT)
        self.device = device
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
        self.to_device()
    def to_device(self):
        self.resnet50_GRU.to(self.device)
    def load(self, model_path):
        self.resnet50_GRU.load_state_dict(torch.load(model_path, map_location=self.device))  
    def set_eval(self):
        self.resnet50_GRU.eval()
        self._is_train = False

    def forward(self, input_image = None):
        assert self._is_train is False, "Model must be in eval mode"
        with torch.no_grad():
            input_image = Variable(input_image)
            input_image = input_image.to(self.device)
            output = self.resnet50_GRU(input_image)
            out_dict = self._format_estimates(output['output'])
            out_dict_raw = dict([(key, output['output'][key]) for key in output['output'].keys()])
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
