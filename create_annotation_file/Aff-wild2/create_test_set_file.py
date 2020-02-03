import pickle
import os
import numpy as np
import argparse
from matplotlib import pyplot as plt
import matplotlib
import glob
import pandas as pd
from tqdm import tqdm
import cv2
parser = argparse.ArgumentParser(description='save annotations')
parser.add_argument('--au_txt', type=str, default = 'aus_test_set.txt')
parser.add_argument('--expr_txt', type=str, default = 'expression_test_set.txt')
parser.add_argument('--va_txt', type=str, default = 'va_test_set.txt')
parser.add_argument('--data_dir', type=str, default= '../cropped_aligned')
parser.add_argument('--video_dir', type=str, default = '../videos')

args = parser.parse_args()

def read_txt(txt_file):
	with open(txt_file, 'r') as f:
		videos = f.readlines()
	videos = [x.strip() for x in videos]
	return videos

def refine_frames_paths(frames, length):
	if (len(frames) > length) and np.abs(len(frames) - length) ==1:
		length = len(frames)
	frames_ids = [int(frame.split('/')[-1].split('.')[0]) - 1 for frame in frames]
	if len(frames) == length:
		return frames
	else:
		extra_frame_ids = []
		prev_frame_id = frames_ids[0]
		for i in range(length):
			if i not in frames_ids:
				extra_frame_ids.append(prev_frame_id)
			else:
				prev_frame_id = i
		frames_ids.extend(extra_frame_ids)
		frames_ids = sorted(frames_ids)
		prefix = '/'.join(frames[0].split('/')[:-1])
		return_frames = [prefix+'/{0:05d}.jpg'.format(id+1) for id in frames_ids]
		return return_frames
def main():
	tasks = ['AU_Set', 'EXPR_Set', 'VA_Set']
	data_file = {}
	for task in tasks:
		data_file[task] = {}
		mode = 'Test_Set'
		txt_file = {'AU_Set':args.au_txt, "EXPR_Set": args.expr_txt, 'VA_Set':args.va_txt}[task]
		videos = read_txt(txt_file)
		data_file[task][mode] = {}
		for video in tqdm(videos):
			name = video
			frames_paths = sorted(glob.glob(os.path.join(args.data_dir, name, '*.jpg')))
			if '_left' in name:
				video_name = name[:-5]
			elif '_right' in name:
				video_name = name[:-6]
			else:
				video_name = name
			video_path = glob.glob(os.path.join(args.video_dir, 'batch*', video_name+".*"))[0]
			cap = cv2.VideoCapture(video_path)
			length = int(cap.get(7)) + 1
			frames_paths = refine_frames_paths(frames_paths, length)
			length = len(frames_paths)
			if task == 'AU_Set':
				AU_list = ['AU1','AU2','AU4','AU6','AU12','AU15','AU20','AU25']
				au_array = np.zeros((length, len(AU_list)))
				data_dict = {'label': au_array}
				data_dict.update({'path': frames_paths, 'frames_ids': np.arange(length)})
			elif task == 'EXPR_Set':
				Expr_list = ['Neutral','Anger','Disgust','Fear','Happiness','Sadness','Surprise']
				data_dict = {'label': np.zeros(length), 'path':frames_paths, 'frames_ids': np.arange(length)}
			elif task == 'VA_Set':
				va_label = np.zeros((length, 2))
				data_dict = {'label':va_label, 'path':frames_paths, 'frames_ids': np.arange(length)}
			data_file[task][mode][name] = data_dict
	save_path = os.path.join('.', 'test_set.pkl')
	pickle.dump(data_file, open(save_path, 'wb'))
if __name__== '__main__':
	main()
