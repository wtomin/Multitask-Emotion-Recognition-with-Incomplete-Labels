import os
import matplotlib.colors as mcolors
import argparse
import matplotlib
matplotlib.use('TkAgg')
font = {'family' : 'normal',
		'size'   : 24}
matplotlib.rc('font', **font)
MODEL_DIR = os.path.abspath('./pytorch-benchmarks/models/')
PRETRAINED_MODEL_DIR = os.path.abspath('./pretrained_models/')
CATEGORIES = {'AU': ['AU1', 'AU2', 'AU4', 'AU6', 'AU12', 'AU15', 'AU20', 'AU25'],
                            'EXPR':['Neutral','Anger','Disgust','Fear','Happiness','Sadness','Surprise'],
                            'VA':['valence', 'arousal']}
Best_AU_Thresholds = {'CNN': [0.1448537, 0.03918985, 0.13766725, 0.02652811, 0.40589422, 0.15572545,0.04808964, 0.10848708],
                      'CNN_RNN': {32: [0.4253935, 0.02641966, 0.1119782, 0.02978198, 0.17256933, 0.06369855, 0.07433069, 0.13828614],
                                  16: [0.30485213, 0.09509478, 0.59577084, 0.4417419, 0.4396544, 0.0452404,0.05204154, 0.0633798 ],
                                  8: [0.4365209 ,0.10177602, 0.2649502,  0.22586018, 0.3772219,  0.07532539, 0.07667687, 0.04306327]}}
tasks = ['AU','EXPR','VA']

__OPT = {"AU_label_size":8 ,
       "EXPR_label_size":7,
       "VA_label_size":2,
       "digitize_num": 20,
       "hidden_size": 128,
       "image_size": 112,
       "tasks": ['EXPR','AU','VA'],
       "pretrained_dataset": 'ferplus'}
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
OPT = AttrDict(__OPT)
# parser = argparse.ArgumentParser(description='Implement the pretrained model on your own data')
# parser.add_argument('--image_dir', type=str, 
#                     help='a directory containing a sequence of cropped and aligned face images')
# parser.add_argument('--model_type', type=str, default='CNN', choices= ['CNN', 'CNN-RNN'],
#                     help='By default, the CNN pretrained models are stored in "Multitask-CNN", and the CNN-RNN \
#                     pretrained models are stored in "Multitask-CNN-RNN"')
# parser.add_argument('--seq_len', type=int, default = 32, choices = [32, 16, 8], help='sequence length when the model type is CNN-RNN')
# parser.add_argument('--image_ext', default = ['.jpg', '.bmp', '.png'], help='image extentions')
# parser.add_argument('--eval_with_teacher', action='store_true', help='whether to predict with teacher model')
# parser.add_argument('--eval_with_students', action='store_true', help='whether to predict with student models')
# parser.add_argument('--ensemble', action='store_true', help='whether to merge the student predictions')
# parser.add_argument('--AU_label_size', type=int, default = 8, help='# of AUs')
# parser.add_argument('--EXPR_label_size', type=int, default = 7, help='# of EXpressions')
# parser.add_argument('--VA_label_size', type=int, default = 2, help='# of VA ')
# parser.add_argument('--digitize_num', type=int, default= 20, choices = [1, 20], help='1 means no digitization,\
#                                                  20 means to digitize continuous label to 20 one hot vector ')
# parser.add_argument('--hidden_size', type=int, default = 128, help='the embedding size of each output head' )
# parser.add_argument('--image_size', type=int, default= 112, help='input image size')
# parser.add_argument('--batch_size', type=int, default= 20, help='input batch size per task')
# parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
# parser.add_argument('--workers', type=int, default=0, help='number of workers')
# parser.add_argument('--tasks', type=str, default = ['EXPR','AU','VA'],nargs="+")
# parser.add_argument('--save_dir', type=str, help='where to save the predictions')
# parser.add_argument('--pretrained_dataset', type=str, default='ferplus',
#                                   choices = ['ferplus', 'sfew','imagenet'], 
#                                   help="the pretrained_dataset of the face feature extractor, choices:['ferplus', 'sfew','imagenet']")
# opt = parser.parse_args()