import pickle
import os
import numpy as np
import argparse
from matplotlib import pyplot as plt
import matplotlib
import glob
import pandas as pd
from tqdm import tqdm
parser = argparse.ArgumentParser(description='save annotations')
parser.add_argument('--vis', action = 'store_true', 
                    help='whether to visualize the distribution')
parser.add_argument('--annot_dir', type=str, default = '/media/Samsung/Aff-wild2-Challenge/annotations',
                    help='annotation dir')
parser.add_argument('--data_dir', type=str, default= '/media/Samsung/Aff-wild2-Challenge/cropped_aligned')

args = parser.parse_args()
def read_AU(txt_file):
	with open(txt_file, 'r') as f:
		lines = f.readlines()
	lines = lines[1:] # skip first line
	lines = [x.strip() for x in lines]
	lines = [x.split(',') for x in lines]
	lines = [[float(y) for y in x ] for x in lines]
	return np.array(lines)
def read_Expr(txt_file):
	with open(txt_file, 'r') as f:
		lines = f.readlines()
	lines = lines[1:] # skip first line
	lines = [x.strip() for x in lines]
	lines = [int(x) for x in lines]
	return np.array(lines)
def read_VA(txt_file):
	with open(txt_file, 'r') as f:
		lines = f.readlines()
	lines = lines[1:] # skip first line
	lines = [x.strip() for x in lines]
	lines = [x.split(',') for x in lines]
	lines = [[float(y) for y in x ] for x in lines]
	return np.array(lines)
def plot_pie(AU_list, pos_freq, neg_freq):
	ploting_labels = [x+'+ {0:.2f}'.format(y) for x, y in zip(AU_list, pos_freq)] + [x+'- {0:.2f}'.format(y) for x, y in zip(AU_list, neg_freq)] 
	cmap = matplotlib.cm.get_cmap('coolwarm')
	colors = [cmap(x) for x in pos_freq] + [cmap(x) for x in neg_freq]
	fracs = np.ones(len(AU_list)*2)
	plt.pie(fracs, labels=ploting_labels, autopct=None, shadow=False, colors=colors,startangle =78.75)
	plt.title("AUs distribution")
	plt.show()

def frames_to_label(label_array, frames, discard_value):
	assert len(label_array) >= len(frames) # some labels need to be discarded
	frames_ids = [int(frame.split('/')[-1].split('.')[0]) - 1 for frame in frames] # frame_id start from 0
	N = label_array.shape[0]
	label_array = label_array.reshape((N, -1))
	to_drop = (label_array == discard_value).sum(-1)
	drop_ids = [i for i in range(len(to_drop)) if to_drop[i]]
	frames_ids = [i for i in frames_ids if i not in drop_ids]
	indexes = [True if i in frames_ids else False for i in range(len(label_array)) ]
	label_array = label_array[indexes]
	assert len(label_array) == len(frames_ids)
	prefix = '/'.join(frames[0].split('/')[:-1])
	return_frames = [prefix+'/{0:05d}.jpg'.format(id+1) for id in frames_ids]
	return label_array, return_frames, frames_ids
def main():
	annot_dir = args.annot_dir
	tasks = [x for x in os.listdir(annot_dir)]
	data_file = {}
	for task in tasks:
		if task == 'AU_Set':
			AU_list = ['AU1','AU2','AU4','AU6','AU12','AU15','AU20','AU25']
			data_file[task] = {}
			for mode in ['Training_Set', 'Validation_Set']:
				txt_files = glob.glob(os.path.join(annot_dir, task, mode, '*.txt'))
				data_file[task][mode] = {}
				for txt_file in tqdm(txt_files):
					name = os.path.basename(txt_file).split('.')[0]
					au_array = read_AU(txt_file)
					frames_paths = sorted(glob.glob(os.path.join(args.data_dir, name, '*.jpg')))
					au_array, frames_paths, frames_ids = frames_to_label(au_array, frames_paths, discard_value = -1)
					data_dict = dict([(AU_list[i], au_array[:, i])for i in range(len(AU_list))])
					data_dict.update({'path': frames_paths, 'frames_ids':frames_ids})
					data_file[task][mode][name] = pd.DataFrame.from_dict(data_dict)
			if args.vis:
				total_dict = {**data_file[task]['Training_Set'], **data_file[task]['Validation_Set']}
				all_samples = []
				for name in total_dict.keys():
					arr = []
					for l in AU_list:
						arr.append(total_dict[name][l].values)
					arr = np.stack(arr, axis=1)
					all_samples.append(arr)
				all_samples = np.concatenate(all_samples, axis=0)
				pos_freq = np.sum(all_samples, axis=0)/all_samples.shape[0]
				neg_freq = -np.sum(all_samples-1, axis=0)/all_samples.shape[0]
				plot_pie(AU_list, pos_freq, neg_freq)
		if task == 'EXPR_Set':
			Expr_list = ['Neutral','Anger','Disgust','Fear','Happiness','Sadness','Surprise']
			data_file[task] = {}
			for mode in ['Training_Set', 'Validation_Set']:
				txt_files = glob.glob(os.path.join(annot_dir, task, mode, '*.txt'))
				data_file[task][mode] = {}
				for txt_file in tqdm(txt_files):
					name = os.path.basename(txt_file).split('.')[0]
					expr_array = read_Expr(txt_file)
					frames_paths = sorted(glob.glob(os.path.join(args.data_dir, name, '*.jpg')))
					expr_array, frames_paths, frames_ids = frames_to_label(expr_array, frames_paths, discard_value = -1)
					data_dict = {'label':expr_array.reshape(-1), 'path':frames_paths, 'frames_ids': frames_ids}
					data_file[task][mode][name] = pd.DataFrame.from_dict(data_dict)
			if args.vis:
				total_dict = {**data_file[task]['Training_Set'], **data_file[task]['Validation_Set']}
				all_samples = np.concatenate([total_dict[x]['label'].values for x in total_dict.keys()], axis=0)
				histogram = np.zeros(len(Expr_list))
				for i in range(len(Expr_list)):
					find_true = sum(all_samples==i)
					histogram[i] =find_true/all_samples.shape[0]
				plt.bar(np.arange(len(Expr_list)), histogram)
				plt.xticks(np.arange(len(Expr_list)), Expr_list)
				plt.show()
		if task == 'VA_Set':
			VA_list = ['Valence', 'Arousal']
			data_file[task] = {}
			for mode in ['Training_Set', 'Validation_Set']:
				txt_files = glob.glob(os.path.join(annot_dir, task, mode, '*.txt'))
				data_file[task][mode] = {}
				for txt_file in tqdm(txt_files):
					name = os.path.basename(txt_file).split('.')[0]
					va_array = read_VA(txt_file) 
					frames_paths = sorted(glob.glob(os.path.join(args.data_dir, name, '*.jpg')))
					va_array, frames_paths, frames_ids = frames_to_label(va_array, frames_paths, discard_value = -5.)
					data_dict = {'valence':va_array[:, 0],'arousal': va_array[:, 1], 'path':frames_paths, 'frames_ids': frames_ids}
					data_file[task][mode][name] = pd.DataFrame.from_dict(data_dict)
			if args.vis:
				total_dict = {**data_file[task]['Training_Set'], **data_file[task]['Validation_Set']}
				all_samples = []
				for name in total_dict.keys():
					arr = []
					for l in ['valence', 'arousal']:
						arr.append(total_dict[name][l].values)
					arr = np.stack(arr, axis=1)
					all_samples.append(arr)
				all_samples = np.concatenate(all_samples, axis=0)
				pos_freq = np.sum(all_samples, axis=0)/all_samples.shape[0]
				plt.hist2d(all_samples[:,0], all_samples[:,1], bins=(20, 20), cmap=plt.cm.jet)
				plt.xlabel("Valence")
				plt.ylabel('Arousal')
				plt.colorbar()
				plt.show()
	save_path = os.path.join(annot_dir, 'annotations.pkl')
	pickle.dump(data_file, open(save_path, 'wb'))
if __name__== '__main__':
	main()
