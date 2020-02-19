import time
from options.test_options import TestOptions
from data.test_video_dataset import Test_dataset
from models.models import ModelsFactory
from collections import OrderedDict
import os
import numpy as np
from sklearn.metrics import f1_score
from PATH import PATH
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from scipy.stats import mode
from scipy.special import softmax
import pickle
from sklearn.metrics import precision_recall_curve
def sigmoid(x):
    return 1/(1+np.exp(-x))
PRESET_VARS = PATH()
#################RuntimeError: received 0 items of ancdata ###########################
import torch
torch.multiprocessing.set_sharing_strategy("file_system")
#########################################################################
class Tester:
    def __init__(self):
        self._opt = TestOptions().parse()
        self._model = ModelsFactory.get_by_name(self._opt.model_name, self._opt)
        test_data_file = PRESET_VARS.Aff_wild2.test_data_file
        self.test_data_file = pickle.load(open(test_data_file, 'rb'))
        self.save_dir = self._opt.save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if self._opt.mode == 'Test':
            self._test()
        else:
            raise ValueError("do not call test.py with validation mode.")
    def _test(self):
        self._model.set_eval()
        val_transforms = self._model.resnet50_GRU.backbone.backbone.compose_transforms
        model_paths = [self._opt.teacher_model_path]
        if self._opt.ensemble:
            for i in range(self._opt.n_students):
                path = os.path.join(self._opt.checkpoints_dir, self._opt.name, 'net_epoch_student_{}_id_resnet50_GRU.pth'.format(i))
                assert os.path.exists(path)
                model_paths.append(path)
        outputs_record = {}
        estimates_record = {}
        frames_ids_record = {}
        for i, path in enumerate(model_paths):
            self._model.resnet50_GRU.load_state_dict(torch.load(path))        
            outputs_record[i] = {}
            estimates_record[i] = {}
            frames_ids_record[i] = {}
            for task in self._opt.tasks:
                task = task+"_Set"
                task_data_file = self.test_data_file[task]['Test_Set']
                outputs_record[i][task] = {}
                estimates_record[i][task] = {}
                frames_ids_record[i][task] = {}
                for i_video, video in enumerate(task_data_file.keys()):
                    video_data = task_data_file[video]
                    test_dataset = Test_dataset(self._opt, video_data, transform=val_transforms)
                    test_dataloader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=self._opt.batch_size,
                    shuffle= False,
                    num_workers=int(self._opt.n_threads_test),
                    drop_last=False)
                    track = self.test_one_video(test_dataloader, task = task[:-4])
                    torch.cuda.empty_cache() 
                    outputs_record[i][task][video] = track['outputs']
                    estimates_record[i][task][video] = track['estimates']
                    frames_ids_record[i][task][video] = track['frames_ids']
                    print("Model ID {} Task {} Current {}/{}".format(i, task[:-4], i_video, len(task_data_file.keys())))
                    save_path = '{}/{}/{}.txt'.format(i, task, video)
                    self.save_to_file(track['frames_ids'], track['estimates'], save_path, task=task[:-4])
                    # if i_video>=1:
                    #     break
        #merge the raw outputs 
        for task in self._opt.tasks:
            task = task+"_Set"
            for video in outputs_record[0][task].keys():
                preds = []
                for i in range(len(outputs_record.keys())):
                    preds.append(outputs_record[i][task][video])
                preds = np.array(preds)
                #assert frames_ids_record[0][task][video] == frames_ids_record[1][task][video]
                video_frames_ids = frames_ids_record[0][task][video] 
                if task == 'AU_Set':
                    merged_preds = sigmoid(preds)
                    best_thresholds_over_models = [] # fill in using the results obtained by val.py
                    print("The best AU thresholds over models: {}".format(best_thresholds_over_models))
                    merged_preds = np.mean(merged_preds, axis=0) 
                    merged_preds = merged_preds > (np.ones_like(merged_preds)*best_thresholds_over_models)
                    merged_preds = merged_preds.astype(np.int64)
                    save_path = '{}/{}/{}.txt'.format('merged', task, video)
                    self.save_to_file(video_frames_ids, merged_preds, save_path, task='AU')
                elif task == 'EXPR_Set':
                    merged_preds = softmax(preds, axis=-1).mean(0).argmax(-1).astype(np.int).squeeze()
                    save_path = '{}/{}/{}.txt'.format('merged',task,  video)
                    self.save_to_file(video_frames_ids, merged_preds, save_path, task='EXPR')
                else:
                    N = self._opt.digitize_num
                    v = softmax(preds[:, :, :N], axis=-1)
                    a = softmax(preds[:, :, N:], axis=-1)
                    bins = np.linspace(-1, 1, num=self._opt.digitize_num)
                    v = (bins * v).sum(-1)
                    a = (bins * a).sum(-1)
                    merged_preds = np.stack([v.mean(0), a.mean(0)], axis = 1).squeeze() 
                    save_path = '{}/{}/{}.txt'.format( 'merged',task, video) 
                    self.save_to_file(video_frames_ids, merged_preds, save_path, task='VA')  

    def save_to_file(self, frames_ids, predictions, save_path, task= 'AU'):
        save_path =os.path.join(self.save_dir, save_path)
        save_dir = os.path.dirname(os.path.abspath(save_path))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        categories = PRESET_VARS.Aff_wild2.categories[task]
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
                        if isinstance(x, float):
                            digits.append("{:.4f}".format(x))
                        elif isinstance(x, np.int64):
                            digits.append(str(x))
                    line = ','.join(digits)+'\n'
                elif isinstance(line, np.int64):
                    line = str(line)+'\n'
                if i == len(predictions)-1:
                    line = line[:-1]
                f.write(line)

    def test_one_video(self, data_loader, task = 'AU'):
        track_val = {'outputs':[], 'estimates':[], 'frames_ids':[]}
        for i_val_batch, val_batch in tqdm(enumerate(data_loader), total = len(data_loader)):
            # evaluate model
            wrapped_v_batch = {task: val_batch}
            self._model.set_input(wrapped_v_batch, input_tasks = [task])
            outputs, _ = self._model.forward(return_estimates=False, input_tasks = [task])
            estimates, _ = self._model.forward(return_estimates=True, input_tasks = [task])
            #store the predictions and labels
            B, N, C = outputs[task][task].shape
            track_val['outputs'].append(outputs[task][task].reshape(B*N, C))
            track_val['frames_ids'].append(np.array([np.array(x) for x in val_batch['frames_ids']]).reshape(B*N, -1).squeeze())
            track_val['estimates'].append(estimates[task][task].reshape(B*N, -1).squeeze())
             
        for key in track_val.keys():
            track_val[key] = np.concatenate(track_val[key], axis=0)
        #assert len(track_val['frames_ids']) -1 == track_val['frames_ids'][-1]
        return track_val
if __name__ == "__main__":
    Tester()
            
