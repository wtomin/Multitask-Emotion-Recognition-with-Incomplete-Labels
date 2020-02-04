import time
from options.test_options import TestOptions
from data.custom_dataset_data_loader import Multitask_DatasetDataLoader
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
import math
import pickle
def sigmoid(x):
    return 1/(1+np.exp(-x))
class Tester:
    def __init__(self):
        self._opt = TestOptions().parse()
        PRESET_VARS = PATH()
        self._model = ModelsFactory.get_by_name(self._opt.model_name, self._opt)
        val_transforms = self._model.resnet50.backbone.compose_transforms
        #self.test_dataloaders = dict([(k, CustomDatasetDataLoader(self._opt, self._opt.mode, self.datasets_names[k], transform = val_transforms).load_data()) for k in self._opt.test_tasks_seq])
        self.validation_dataloaders = Multitask_DatasetDataLoader(self._opt, train_mode = self._opt.mode, transform = val_transforms)
        self.validation_dataloaders = self.validation_dataloaders.load_multitask_val_test_data()
        print("{} sets".format(self._opt.mode))
        for task in self._opt.tasks:
            data_loader = self.validation_dataloaders[task]
            print("{}: {} images".format(task, len(data_loader)*self._opt.batch_size * len(self._opt.tasks)))
        if self._opt.mode == 'Validation':
            self._validate()
        else:
            raise ValueError("do not call val.py with test mode.")
        
    def _validate(self):
        val_start_time = time.time()
        # set model to eval
        self._model.set_eval()
        model_paths = [self._opt.teacher_model_path]
        if self._opt.ensemble:
            for i in range(self._opt.n_students):
                path = os.path.join(self._opt.checkpoints_dir, self._opt.name, 'net_epoch_student_{}_id_resnet50.pth'.format(i))
                assert os.path.exists(path)
                model_paths.append(path)
        outputs_record = {}
        estimates_record = {}
        metrics_record = {}
        labels_record = {}
        for i, path in enumerate(model_paths):
            self._model.resnet50.load_state_dict(torch.load(path))
            outputs_record[i] = {}
            estimates_record[i] = {}
            metrics_record[i] = {}
            labels_record[i] = {}
            for task in self._opt.tasks:
                track_val = {'outputs':[],'labels':[], 'estimates':[]}
                data_loader = self.validation_dataloaders[task]
                for i_val_batch, val_batch in tqdm(enumerate(data_loader), total = len(data_loader)):
                    # evaluate model
                    wrapped_v_batch = {task: val_batch}
                    self._model.set_input(wrapped_v_batch, input_tasks = [task])
                    outputs, _ = self._model.forward(return_estimates=False, input_tasks = [task])
                    estimates, _ = self._model.forward(return_estimates=True, input_tasks = [task])
                    #store the predictions and labels
                    track_val['outputs'].append(outputs[task][task])
                    track_val['labels'].append(wrapped_v_batch[task]['label'])
                    track_val['estimates'].append(estimates[task][task])
                    ##if i_val_batch>10:
                        #break
                # calculate metric
                for key in track_val.keys():
                    track_val[key] = np.concatenate(track_val[key], axis=0)
                preds = track_val['estimates']
                labels = track_val['labels']
                metric_func = self._model.get_metrics_per_task()[task]
                eval_items, eval_res = metric_func(preds, labels)
                now_time = time.strftime("%H:%M", time.localtime(val_start_time))
                output = "Model id {} {} Validation {}: Eval_0 {:.4f} Eval_1 {:.4f} eval_res {:.4f}".format(i, task, 
                    now_time, eval_items[0], eval_items[1], eval_res)
                print(output)
                outputs_record[i][task] = track_val['outputs']
                estimates_record[i][task] = track_val['estimates']
                labels_record[i][task] = track_val['labels']
                metrics_record[i][task] = [eval_items, eval_res]
        # one choice, merge the estimates
        for task in self._opt.tasks:
            preds = []
            labels = []
            for i in range(len(estimates_record.keys())):
                preds.append(estimates_record[i][task])
                labels.append(labels_record[i][task])
            preds = np.array(preds)
            labels = np.array(labels)
            #assert labels[0] == labels[1]
            if task == 'AU' or task == 'EXPR':
                merged_preds = mode(preds, axis=0)[0]
            elif task == 'VA':
                merged_preds = np.mean(preds, axis=0)
            labels = np.mean(labels,axis=0)
            #assert labels.shape[0] == merged_preds.shape[0]
            metric_func = self._model.get_metrics_per_task()[task]
            eval_items, eval_res = metric_func(merged_preds.squeeze(), labels.squeeze())
            now_time = time.strftime("%H:%M", time.localtime(val_start_time))
            output = "Merged First method {} Validation {}: Eval_0 {:.4f} Eval_1 {:.4f} eval_res {:.4f}".format( task, 
                now_time, eval_items[0], eval_items[1], eval_res)
            print(output)
        # one choice, average the raw outputs, this one is better than the first one
        for task in self._opt.tasks:
            preds = []
            labels = []
            for i in range(len(estimates_record.keys())):
                preds.append(outputs_record[i][task])
                labels.append(labels_record[i][task])
            preds = np.array(preds)
            labels = np.array(labels)
            #assert labels[0] == labels[1]
            if task == 'AU':
                merged_preds = (sigmoid(preds)>0.5).astype(np.int)
                merged_preds = mode(merged_preds, axis=0)[0]
            elif task=='EXPR':
                merged_preds = softmax(preds, axis=-1).mean(0).argmax(-1).astype(np.int)
            else:
                N = self._opt.digitize_num
                v = softmax(preds[:, :, :N], axis=-1)
                a = softmax(preds[:, :, N:], axis=-1)
                bins = np.linspace(-1, 1, num=self._opt.digitize_num)
                v = (bins * v).sum(-1)
                a = (bins * a).sum(-1)
                merged_preds = np.stack([v.mean(0), a.mean(0)], axis = 1)
            labels = np.mean(labels, axis=0)
            metric_func = self._model.get_metrics_per_task()[task]
            eval_items, eval_res = metric_func(merged_preds.squeeze(), labels.squeeze())
            now_time = time.strftime("%H:%M", time.localtime(val_start_time))
            output = "Merged Second method {} Validation {}: Eval_0 {:.4f} Eval_1 {:.4f} eval_res {:.4f}".format(task, 
                now_time, eval_items[0], eval_items[1], eval_res)
            print(output)
        save_path = 'evaluate_val_set.pkl'
        data = {'outputs':outputs_record, 'estimates':estimates_record, 'labels':labels_record, 'metrics':metrics_record}
        pickle.dump(data, open(save_path, 'wb'))

if __name__ == "__main__":
    Tester()
