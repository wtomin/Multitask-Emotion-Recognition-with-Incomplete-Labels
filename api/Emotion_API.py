import cv2
from tqdm import tqdm
from models.ModelFactory import ModelFactory
from data.Seq_Dataset import Seq_Dataset
from data.Image_Dataset import Image_Dataset
from config import tasks as TASKS
from config import Best_AU_Thresholds, OPT

class Emotion_API_Video(object):
    # given a inptu video file return a csv file containing all three types of emotion predictions for all available frames.
    def __init__(self, 
        device = None,
        use_temporal = False, # Use CNN model by default
        num_students = 1, # Use only 1 student model. One can use at most five student models, which can cause more computation cost.
        video_output = None,
        OpenFace_exe = 'OpenFace/build/bin/FeatureExtraction',
        # parameters for OpenFace feature extraction, passed to Video_Processor
        save_size=112, nomask=True, grey=False, quiet=True,
        tracked_vid=False, noface_save=False,
        # parameters for image sampling, passed to image sampler
        length = 32, 
        num_workers = 0, 
        # minimum frames allowed
        min_frames = 1, 
        batch_size = 24
        # whether to save csv file
        save_csv = True,
        # whether to save annotated video
        emotion_annotated_video=False
        ):
        self.video_file = video_file
        self.num_workers = num_workers
        self.batch_size = batch_size
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.video_processor = Video_Processor(save_size, nomask, grey, quiet,
                              tracked_vid, noface_save, OpenFace_exe)
        self.model_type = 'CNN' if not use_temporal else 'CNN_RNN'
        self.ensemble, self.val_transforms = ModelFactory.get(model_type, num_students)
        self.save_csv = save_csv
        self.emotion_annotated_video = emotion_annotated_video
    def run(self, video_file , save_path = None):
        video_cap = cv2.VideoCapture(video_file)
        # self.fps = int(round(self.video.get(cv2.CAP_PROP_FPS)))
        # self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        video_name = os.path.basename(video_file).split('.')[0]
        # first input video is processed using OpenFace
        opface_output_dir = os.path.join(os.path.dirname(video_file), 
                video_name+"_opface")
        self.video_processor.process(video_file, opface_output_dir)
        assert len(os.listdir(opface_output_dir)) >0 , "The OpenFace output directory should not be empty: {}".format(opface_output_dir)
        # sample images
        if self.model_type == 'CNN_RNN':
            dataset = Seq_Dataset(opface_output_dir, 
                seq_len=self.length, transform = self.val_transforms)
        else:
            dataset = Image_Dataset(opface_output_dir,
                transform = self.val_transforms)
        total_frames = len(dataset)
        assert total_frames> self.min_frames,"The minimum number of frames should be {}".format(self.min_frames)
        dataloader =  torch.utils.data.DataLoader(
                        dataset,
                        batch_size=self.batch_size,
                        shuffle= False,
                        num_workers=self.workers,
                        drop_last=False)

        ensemble_estimates = self.test_ensemble(dataloader)
        if self.save_csv:
        	if save_path is None:
                save_path = os.path.join(os.path.dirname(video_file), '{}.csv'.format(video_name))
            
            self.save_preds_to_file(ensemble_estimates, save_path)
        # if self.emotion_annotated_video:
        #     save_path = os.path.join(os.path.dirname(video_file), '{}_emotion.avi'.format(video_name))
        #     self.save_preds_to_video(video_cap, ensemble_estimates, save_path)

    # def save_preds_to_video(self, video_cap, preds_dict, save_path):
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #     fps = int(round(video.get(cv2.CAP_PROP_FPS)))
    #     out_video = cv2.VideoWriter(filename=save_path, 
    #         fourcc=fourcc, 
    #         frameSize=(1920, 1080),
    #         fps=fps)
    #     success,frame = video_cap.read()
    #     i_frame = 0
    #     while success:
    #     	for task in TASKS:
    #     		pred_task = preds_dict[task][i_frame]


    def save_preds_to_file(self, preds_dict, save_path):
        df= pd.DataFrame.from_dict(preds_dict)
        df.to_csv(save_path)
    def test_single_model(self,
        model, data_loader):
        track_val = {}
        for task in TASKS:
            track_val[task] = {'outputs':[],'frames_ids':[]}
        model.set_eval()
        with torch.no_grad():
            for i_val_batch, val_batch in tqdm(enumerate(data_loader), total = len(data_loader)):
                estimates, outputs = model.forward( input_image = val_batch['image'])
                #store the predictions and labels
                for task in TASKS:
                    if 'RNN' in self.model_type:
                        B, N, C = outputs[task].shape
                        track_val[task]['outputs'].append(outputs[task].reshape(B*N, C))
                        track_val[task]['frames_ids'].append(np.array([np.array(x) for x in val_batch['frames_ids']]).reshape(B*N, -1).squeeze())
                        #track_val[task]['estimates'].append(estimates[task].reshape(B*N, -1).squeeze())
                    else:
                        track_val[task]['outputs'].append(outputs[task])
                        track_val[task]['frames_ids'].append(np.array(val_batch['frames_ids']))
                        #track_val[task]['estimates'].append(estimates[task])

        for task in TASKS:
            for key in track_val[task].keys():
                track_val[task][key] = torch.cat(track_val[task][key], dim=0)
        assert len(track_val['frames_ids']) -1 == track_val['frames_ids'][-1]
        return track_val
    
    def test_ensemble(self, dataloader):
        models_preds = {}
        for i_model, model in enumerate(self.ensemble):
            preds = self.test_single_model(model, dataloader)
            models_preds[i_model] = preds
        ensemble_outputs = {}
        frames_ids = models_preds[0][TASKS[0]]['frames_ids']
        for task in TASKS:
            ensemble_outputs[task] = torch.stack([models_preds[i_model][task]['outputs'] for i_model in range(len(self.ensemble))], dim=0).mean(0)
            assert all([models_preds[i_model][task]['frames_ids']==frames_ids for i_model in range(len(self.ensemble))])
        return self._format_estimates(ensemble_outputs)

    def _format_estimates(self, output):
        estimates = {}
        for task in output.keys():
            if task == 'AU':
                best_au_thresholds = Best_AU_Thresholds[self.model_type]
                if 'RNN' in self.model_type:
                    best_au_thresholds = best_au_thresholds[self.length]
                
                o = torch.sigmoid(output['AU'].cpu())
                threshold = torch.ones_like(o) * torch.Tensor(best_au_thresholds)
                o = (o>best_au_thresholds).type(torch.LongTensor)              
                estimates['AU'] = o.numpy()
            elif task == 'EXPR':
                o = F.softmax(output['EXPR'].cpu(), dim=-1).argmax(-1).type(torch.LongTensor)
                estimates['EXPR'] = o.numpy()
            elif task == 'VA':
                N = OPT.digitize_num
                v = F.softmax(output['VA'][:, :N].cpu(), dim=-1).numpy()
                a = F.softmax(output['VA'][:, N:].cpu(), dim=-1).numpy()
                bins = np.linspace(-1, 1, num=OPT.digitize_num)
                v = (bins * v).sum(-1)
                a = (bins * a).sum(-1)
                estimates['VA'] = np.stack([v, a], axis = 1)
        return estimates

