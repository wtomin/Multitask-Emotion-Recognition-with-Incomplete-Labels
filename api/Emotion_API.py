import cv2
import os
import torch
from video_processor import Video_Processor
from tqdm import tqdm
from models.ModelFactory import ModelFactory
from data.Seq_Dataset import Seq_Dataset
from data.Image_Dataset import Image_Dataset
from config import tasks as TASKS
from config import Best_AU_Thresholds, OPT, CATEGORIES
import numpy as np
import torch.nn.functional as F
import pandas as pd
from scipy.stats import mode

class Emotion_API(object):
    # given a inptu video file return a csv file containing all three types of emotion predictions for all available frames.
    def __init__(self, 
        device = None,
        use_temporal = False, # Use CNN model by default
        num_students = 1, # Use only 1 student model. One can use at most five student models, which can cause more computation cost.
        OpenFace_exe = './OpenFace/build/bin/FeatureExtraction',
        # parameters for OpenFace feature extraction, passed to Video_Processor
        save_size=112, nomask=True, grey=False, quiet=True,
        tracked_vid=False, noface_save=False,
        # parameters for image sampling, passed to image sampler
        length = 32, 
        num_workers = 0, 
        # minimum frames allowed
        min_frames = 1, 
        batch_size = 24,
        # whether to save csv file
        save_csv = True,
        # whether to save annotated video
        emotion_annotated_video=False
        ):

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.length = length
        self.min_frames = min_frames
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print("Using device :{}".format(device))
        self.video_processor = Video_Processor(save_size, nomask, grey, quiet,
                              tracked_vid, noface_save, OpenFace_exe)
        self.model_type = 'CNN' if not use_temporal else 'CNN_RNN'
        self.ensemble, self.val_transforms = ModelFactory.get(self.device, self.model_type, num_students)
        self.save_csv = save_csv
        self.emotion_annotated_video = emotion_annotated_video
    def run_face_images(self, face_dir, csv_output=None):
        # face_dir is a directory containing a sequence of cropped and aligned face images
        assert len(os.listdir(face_dir))>0, "{} is empty".format(face_dir)
        if self.model_type == 'CNN_RNN':
            dataset = Seq_Dataset(face_dir, 
                seq_len=self.length, transform = self.val_transforms)
            total_frames = len(dataset) * self.length
        else:
            dataset = Image_Dataset(face_dir,
                transform = self.val_transforms)
            total_frames = len(dataset) 
        
        assert total_frames> self.min_frames,"The minimum number of frames should be {}".format(self.min_frames)
        dataloader =  torch.utils.data.DataLoader(
                        dataset,
                        batch_size=self.batch_size,
                        shuffle= False,
                        num_workers=self.num_workers,
                        drop_last=False)

        
        ensemble_estimates = self.test_ensemble(dataloader)
        if self.save_csv:
            if csv_output is None:
                csv_output = os.path.join(os.path.dirname(video_file), '{}.csv'.format(video_name))
            
            self.save_preds_to_file(ensemble_estimates, csv_output)
    def run_raw_images(self, images_dir, csv_output=None):
        # images_dir is a directory containing a sequence of video frames. Each frame may contain faces.
        assert len(os.listdir(images_dir))>0, "The images_dir is empty!"

        opface_output_dir = os.path.join(os.path.dirname(images_dir), 
                os.path.basename(images_dir)+"_opface")
        assert not os.path.exists(opface_output_dir), "{} exists before OpenFace is extracting".format(opface_output_dir)
        self.video_processor.process(images_dir, opface_output_dir)
        assert len(os.listdir(opface_output_dir)) >0 , "The OpenFace output directory should not be empty: {}".format(opface_output_dir)
        aligned_face_dir = os.path.join(opface_output_dir, '{}_aligned'.format(os.path.basename(images_dir)))
        if os.path.exists(aligned_face_dir):
            self.run_face_images(aligned_face_dir, csv_output)
        else:
            self.run_face_images(opface_output_dir, csv_output)
        

    def run(self, video_file , csv_output = None):
        video_cap = cv2.VideoCapture(video_file)
        video_name = os.path.basename(video_file).split('.')[0]
        # first input video is processed using OpenFace
        opface_output_dir = os.path.join(os.path.dirname(video_file), 
                video_name+"_opface")
        if not os.path.exists(opface_output_dir):
            self.video_processor.process(video_file, opface_output_dir)
        assert len(os.listdir(opface_output_dir)) >0 , "The OpenFace output directory should not be empty: {}".format(opface_output_dir)
        
        # sample images
        face_dir = os.path.join(opface_output_dir, '{}_aligned'.format(video_name))
        if self.model_type == 'CNN_RNN':
            dataset = Seq_Dataset(face_dir, 
                seq_len=self.length, transform = self.val_transforms)
            total_frames = len(dataset) * self.length
        else:
            dataset = Image_Dataset(face_dir,
                transform = self.val_transforms)
            total_frames = len(dataset) 
        
        assert total_frames> self.min_frames,"The minimum number of frames should be {}".format(self.min_frames)
        dataloader =  torch.utils.data.DataLoader(
                        dataset,
                        batch_size=self.batch_size,
                        shuffle= False,
                        num_workers=self.num_workers,
                        drop_last=False)

        
        ensemble_estimates = self.test_ensemble(dataloader)
        if self.save_csv:
            if csv_output is None:
                csv_output = os.path.join(os.path.dirname(video_file), '{}.csv'.format(video_name))
            
            self.save_preds_to_file(ensemble_estimates, csv_output)
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
        frames_ids = preds_dict['frames_ids']
        unique_frames_ids = sorted(np.unique(frames_ids))
        df = {'frames_ids':[]}
        for frame_id in unique_frames_ids:
            ids_mask = np.array([f_id == frame_id for f_id in frames_ids])
            df['frames_ids'].append(frame_id)
            for task in TASKS:
                categories = CATEGORIES[task]
                if task != 'EXPR':
                    for i_c, cate in enumerate(categories):
                        if cate not in df.keys():
                            df[cate] = []
                        if task == 'VA':
                            df[cate].append(preds_dict[task][ids_mask, i_c].mean(0))
                        else:
                            df[cate].append(mode(preds_dict[task][ids_mask, i_c]).mode[0])
                else:
                    if task not in df.keys():
                        df[task] = []
                    df[task].append(mode(preds_dict[task][ids_mask]).mode[0])
        df = pd.DataFrame.from_dict(df)
        df.to_csv(save_path, index=False)

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
                try:
                    track_val[task][key] = torch.cat(track_val[task][key], dim=0)
                except TypeError:
                    track_val[task][key] = np.concatenate(track_val[task][key], axis=0)
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
            assert all([(models_preds[i_model][task]['frames_ids']==frames_ids).all() for i_model in range(len(self.ensemble))])
        ensemble_outputs = self._format_estimates(ensemble_outputs)
        ensemble_outputs.update({"frames_ids": frames_ids})
        return ensemble_outputs

    def _format_estimates(self, output):
        estimates = {}
        for task in output.keys():
            if task == 'AU':
                best_au_thresholds = Best_AU_Thresholds[self.model_type]
                if 'RNN' in self.model_type:
                    best_au_thresholds = best_au_thresholds[32] # since the downloaded models have trained with sequence length=32
                
                o = torch.sigmoid(output['AU'].cpu())
                threshold = torch.ones_like(o) * torch.Tensor(best_au_thresholds)
                o = (o>threshold).type(torch.LongTensor)              
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


