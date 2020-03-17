import argparse
import os
import torch

class BaseOptions():
    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self._initialized = False

    def initialize(self):
        self._parser.add_argument('--load_epoch', type=int, default=-1, help='which epoch to load? set to -1 to use latest cached model')
        self._parser.add_argument('--temperature', type=float, default=1.5, help='temperature in distillation loss')
        self._parser.add_argument('--AU_label_size', type=int, default = 8, help='# of AUs')
        self._parser.add_argument('--EXPR_label_size', type=int, default = 7, help='# of EXpressions')
        self._parser.add_argument('--VA_label_size', type=int, default = 2, help='# of VA ')
        self._parser.add_argument('--digitize_num', type=int, default= 20, choices = [1, 20], help='1 means no digitization,\
                                                         20 means to digitize continuous label to 20 one hot vector ')
        self._parser.add_argument('--AU_criterion', type=str, default = 'BCE', choices = ['FocalLoss', 'BCE'])
        self._parser.add_argument('--EXPR_criterion', type=str, default = 'CE', choices = ['FocalLoss', 'CE'])
        self._parser.add_argument('--VA_criterion', type=str, default = 'CCC_CE', choices = ['CCC', 'CCC_CE', 'CCC_FocalLoss'])
        self._parser.add_argument('--lambda_teacher', type=float, default = 0.4, help='weight for distillation loss when the ground truth exists (between 0 to 1)')
        self._parser.add_argument('--lambda_AU', type=float, default= 8., help='weight for AU.')
        self._parser.add_argument('--lambda_EXPR', type=float, default= 1., help='weight for EXPR.')
        self._parser.add_argument('--lambda_V', type=float, default= 1., help='weight for valence.')
        self._parser.add_argument('--lambda_A', type=float, default= 1., help='weight for arousal.')
        self._parser.add_argument('--lambda_ccc', type=float, default= 1., help='weight for ccc loss in (CE + lambda_ccc*ccc).')
        #self._parser.add_argument('--force_balance', action='store_true', help='force data balanced for training set')
        self._parser.add_argument('--dataset_names', type=str, default = ['Mixed_EXPR','Mixed_AU','Mixed_VA'],nargs="+")
        self._parser.add_argument('--tasks', type=str, default = ['EXPR','AU','VA'],nargs="+")
        # 'dataset_names' need to be in the same order as the 'tasks'
        self._parser.add_argument('--seq_len', type=int, default=64, help='length of input seq ')
        self._parser.add_argument('--frozen', action='store_true')
        self._parser.add_argument('--hidden_size', type=int, default = 128, help='the embedding size of each output head' )
        self._parser.add_argument('--batch_size', type=int, default= 20, help='input batch size per task')
        self._parser.add_argument('--image_size', type=int, default= 224, help='input image size') # reducing iamge size is acceptable
        self._parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self._parser.add_argument('--name', type=str, default='experiment_1', help='name of the experiment. It decides where to store samples and models')
        self._parser.add_argument('--n_threads_train', default=8, type=int, help='# threads for loading data')
        self._parser.add_argument('--n_threads_test', default=8, type=int, help='# threads for loading data')
        self._parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self._parser.add_argument('--loggings_dir', type=str, default='./loggings', help='loggings are saved here')
        self._parser.add_argument('--model_name', type=str, default='resnet50', help='the name of model')
        self._parser.add_argument('--pretrained_dataset', type=str, default='ferplus',
                                  choices = ['ferplus', 'sfew','imagenet'], 
                                  help="the pretrained_dataset of the face feature extractor, choices:['ferplus', 'sfew','imagenet']")

        self._parser.add_argument('--pretrained_resnet50_model', type=str, default = '', help='pretrained model')
        self._parser.add_argument('--pretrained_teacher_model', type=str, default='')
        self._initialized = True

    def parse(self):
        if not self._initialized:
            self.initialize()
        self._opt = self._parser.parse_args()

        # set is train or test
        self._opt.is_train = self.is_train

        # set and check load_epoch
        self._set_and_check_load_epoch()

        # get and set gpus
        self._get_set_gpus()

        args = vars(self._opt)

        # print in terminal args
        self._print(args)

        # save args to file
        self._save(args)

        return self._opt

    def _set_and_check_load_epoch(self):
        models_dir = os.path.join(self._opt.checkpoints_dir, self._opt.name)
        if os.path.exists(models_dir):
            if self._opt.load_epoch == -1:
                load_epoch = 0
                if self.is_train:
                    for file in os.listdir(models_dir):
                        if file.startswith("net_epoch_"):
                            load_epoch = max(load_epoch, int(file.split('_')[2]))
                self._opt.load_epoch = load_epoch
            else:
                found = False
                for file in os.listdir(models_dir):
                    if file.startswith("net_epoch_"):
                        found = int(file.split('_')[2]) == self._opt.load_epoch
                        if found: break
                assert found, 'Model for epoch %i not found' % self._opt.load_epoch
        else:
            assert self._opt.load_epoch < 1, 'Model for epoch %i not found' % self._opt.load_epoch
            self._opt.load_epoch = 0

    def _get_set_gpus(self):
        # get gpu ids
        str_ids = self._opt.gpu_ids.split(',')
        self._opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self._opt.gpu_ids.append(id)

        # set gpu ids
        if len(self._opt.gpu_ids) > 0:
            torch.cuda.set_device(self._opt.gpu_ids[0])

    def _print(self, args):
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

    def _save(self, args):
        expr_dir = os.path.join(self._opt.checkpoints_dir, self._opt.name)
        print(expr_dir)
        if self.is_train:
            os.makedirs(expr_dir)
        else:
            assert os.path.exists(expr_dir)
        file_name = os.path.join(expr_dir, 'opt_%s.txt' % ('train' if self.is_train else 'test'))
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
