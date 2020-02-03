import time
from options.test_options import TestOptions
from data.custom_dataset_data_loader import CustomDatasetDataLoader
from models.models import ModelsFactory
from collections import OrderedDict
import os
import numpy as np
from sklearn.metrics import f1_score
from PATH import PATH
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
class Tester:
    def __init__(self):
        self._opt = TestOptions().parse()
        PRESET_VARS = PATH()
        self._model = ModelsFactory.get_by_name(self._opt.model_name, self._opt)
        self.has_label = True 
        self.datasets_names = {"AU": self._opt.AU_dataset, "EXPR": self._opt.EXPR_dataset, 'VA':self._opt.VA_dataset}
        val_transforms = self._model.current_model.backbone.compose_transforms
        self.test_dataloaders = dict([(k, CustomDatasetDataLoader(self._opt, 'Validation', self.datasets_names[k], transform = val_transforms).load_data()) for k in self._opt.test_tasks_seq])
        print("Tasks\tDatasets\tTest Size(# of images)")
        for k in self._opt.test_tasks_seq:
            print("{}\t{}\t{}".format(k, self.datasets_names[k], 
                len(self.test_dataloaders[k]) * self._opt.batch_size))
        self._test()
    def _test(self):
        test_start_time = time.time()
        self._model.set_eval()
        dicts = {}
        for task in self._opt.test_tasks_seq:
            print("Evaluate the current model in {} dataset".format(self.datasets_names[task]))
            track_preds = []
            track_labels = []
            track_indexes = []
            track_paths = []
            for i_test_batch, test_batch in tqdm(enumerate(self.test_dataloaders[task]), total = len(self.test_dataloaders[task])):
                outputs = self._model.inference_current(test_batch, task, return_estimates=True)
                label = test_batch['label']
                index = test_batch['index']
                pred = outputs[task]
                track_labels.append(deepcopy(label))
                track_preds.append(deepcopy(pred))
                track_indexes.append(deepcopy(index))
                del test_batch, outputs
            preds = np.concatenate(track_preds, axis=0)
            labels = np.concatenate(track_labels, axis=0)
            track_indexes = np.concatenate(track_indexes, axis=0)
            data = dict([('{}'.format(task), preds), ('index',track_indexes)])
            if self.has_label:
                metric = self._model.get_metrics_per_task()[task]
                eval_item, eval_res = metric(preds, labels)
                data.update({'metric':eval_res, 'metric_0': eval_item[0], 'metric_1': eval_item[1]})
                print("Evaluation Performanace for {}: res {}, res_0 {}, res_1 {}".format(task, eval_res, eval_item[0], eval_item[1]))
            dicts[task] = data
        save_path = os.path.join(self._opt.loggings_dir, self._opt.save_path)
        pickle.dump(dicts, open(save_path, 'wb'))


if __name__ == "__main__":
    Tester()