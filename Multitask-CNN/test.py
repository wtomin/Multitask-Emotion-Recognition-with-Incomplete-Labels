import time
from options.test_options import TestOptions
from data.test_video_dataset import Test_dataset
from models.models import ModelsFactory
from collections import OrderedDict
import os
import numpy as np
import torch
from sklearn.metrics import f1_score
from PATH import PATH
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from scipy.stats import mode
from scipy.special import softmax

class Tester:
    def __init__(self):
        self._opt = TestOptions().parse()
        PRESET_VARS = PATH()
        self._model = ModelsFactory.get_by_name(self._opt.model_name, self._opt)
        test_data_file = PRESET_VARS.Aff_wild2.test_data_file
        self.test_data_file = pickle.load(open(test_data_file, 'rb'))
    def _test(self):
        self._model.set_eval()
        val_transforms = self._model.resnet50.backbone.compose_transforms
        model_paths = [self._opt.teacher_model_path]
        if self._opt.ensemble:
            for i in range(self._opt.n_students):
                path = os.path.join(self._opt.checkpoints_dir, self._opt.name, 'net_epoch_student_{}_id_resnet50.pth'.format(i))
                assert os.path.exists(path)
                model_paths.append(path)
        outputs_record = {}
        estimates_record = {}
        frames_ids_record = {}
        for i, path in enumerate(model_paths):
            self._model.resnet50.load_state_dict(torch.load(path))   
                     
        outputs_record = {}
        estimates_record = {}
        frames_ids_record = {}
            
            for task in self._opt.tasks:
                task_data_file = self.test_data_file[task]['Test_Set']
                
                for video in task_data_file.keys():
                    video_data = task_data_file[video]
                    test_dataset = Test_dataset(self._opt, video_data, transform=val_transforms)
                    test_dataloader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=self._opt.batch_size,
                    shuffle= False,
                    num_workers=int(self._opt.n_threads_test),
                    drop_last=False)
                    track = self.test_one_video(test_dataloader)
                    outputs_record[i][task][video] = track['outputs']
                    estimates_record[i][task][video] = track['estimates']
                    frames_ids_record[i][task][video] = track['frames_ids']
                    
                    
                
     def test_one_video(self, data_loader):
         track_val = {'outputs':[], 'estimates':[], 'frames_ids':[]}
         
         for i_val_batch, val_batch in tqdm(enumerate(data_loader), total = len(data_loader)):
              # evaluate model
             wrapped_v_batch = {task: val_batch}
             self._model.set_input(wrapped_v_batch, input_tasks = [task])
             outputs, _ = self._model.forward(return_estimates=False, input_tasks = [task])
             estimates, _ = self._model.forward(return_estimates=True, input_tasks = [task])
             #store the predictions and labels
             track_val['outputs'].append(outputs[task][task])
             track_val['frames_ids'].append(np.array(val_batch['frames_ids']))
             track_val['estimates'].append(estimates[task][task])
             
         for key in track_val.keys():
             track_val[key] = np.concatenate(track_val[key], axis=0)
         ids = np.argsort(track_val['frames_ids'])
         for key in track_val.keys():
             track_val[key] = track_val[key][ids]
         return track_val
         
         
            
