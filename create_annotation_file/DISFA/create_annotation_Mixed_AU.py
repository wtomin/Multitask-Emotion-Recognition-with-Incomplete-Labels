import pickle
import os
import numpy as np
import argparse
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 22})
import glob
import pandas as pd
from PIL import Image
from tqdm import tqdm
import random
parser = argparse.ArgumentParser(description='read two annotations files')
parser.add_argument('--vis', action='store_true')
parser.add_argument('--aff_wild2_pkl', type=str, default = '/media/Samsung/Aff-wild2-Challenge/annotations/annotations.pkl')
parser.add_argument('--DISFA_pkl', type=str, default = '/media/Samsung/DISFA/annotations.pkl')
parser.add_argument('--save_path', type=str, default='/media/Samsung/Aff-wild2-Challenge/exps/create_new_training_set_AU/create_annotation_file/mixed_AU_annotations.pkl')
args = parser.parse_args()
AU_list = ['AU1','AU2','AU4','AU6','AU12','AU15','AU20','AU25']
def read_data(data_dict):
	paths = []
	labels = []
	for video in data_dict.keys():
		df = data_dict[video]
		labels.append(df[AU_list].values.astype(np.float32))
		paths.append(df['path'].values)
	paths = np.concatenate(paths, axis=0)
	labels = np.concatenate(labels, axis=0)
	return {'label': labels, 'path': paths}
def read_aff_wild2():
	total_data = pickle.load(open(args.aff_wild2_pkl, 'rb'))
	# training set
	data = total_data['AU_Set']['Training_Set']
	data = read_data(data)
	# validation set
	val_data = total_data['AU_Set']['Validation_Set']
	val_data = read_data(val_data)
	return data, val_data
def merge_two_datasets():
	data_aff_wild2, data_aff_wild2_val = read_aff_wild2()
	data_DISFA = pickle.load(open(args.DISFA_pkl, 'rb'))
	data_DISFA = {**data_DISFA['Training_Set'], **data_DISFA['Validation_Set']}
	paths = []
	labels = []
	for video in data_DISFA.keys():
		data = data_DISFA[video]
		labels.append(data['label'])
		paths.append(data['path'])
	paths = np.concatenate(paths, axis=0)
	labels = np.concatenate(labels, axis=0)
	data_DISFA = {'label': labels, 'path': paths}
	data_merged = {'label': np.concatenate((data_aff_wild2['label'], data_DISFA['label']), axis=0),
					'path': list(data_aff_wild2['path']) + list(data_DISFA['path'])}
	if args.vis:
		plot_n_samples_each_cate(data_aff_wild2['label'])
		plot_n_samples_each_cate(data_DISFA['label'])
		plot_n_labels_each_instance(data_aff_wild2['label'])
		plot_n_labels_each_instance(data_DISFA['label'])
	return {'Training_Set': data_merged, 'Validation_Set': data_aff_wild2_val}
def autolabel(rects, ax, bar_label):
	for idx,rect in enumerate(rects):
		height = rect.get_height()
		ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
				bar_label[idx],
				ha='center', va='bottom', rotation=0)
def plot_n_samples_each_cate(labels):
	fig, ax = plt.subplots()
	pos_samples = np.sum(labels, axis=0)
	bar_plot = plt.bar(np.arange(len(AU_list)), pos_samples)
	autolabel(bar_plot, ax, [str(x) for x in pos_samples])
	plt.xticks(np.arange(len(AU_list)), AU_list)
	plt.ylabel("Number of Samples")
	plt.show()
def plot_n_labels_each_instance(labels):
	fig, ax = plt.subplots()
	pos_samples = np.sum(labels, axis=1)
	unique_nums = np.unique(pos_samples)
	count_nums = [sum(pos_samples==i)for i in unique_nums]
	bar_plot = plt.bar(np.arange(len(unique_nums)), count_nums)
	autolabel(bar_plot, ax, [str(x) for x in count_nums])
	plt.xticks(np.arange(len(unique_nums)), unique_nums)
	plt.ylabel("Number of Samples")
	plt.xlabel("Number of Labels")
	plt.show()
def IRLbl(labels):
	# imbalance ratio per label
	# Args:
	#	 labels is a 2d numpy array, each row is one instance, each column is one class; the array contains (0, 1) only
	N, C = labels.shape
	pos_nums_per_label = np.sum(labels, axis=0)
	max_pos_nums = np.max(pos_nums_per_label)
	return max_pos_nums/pos_nums_per_label

def MeanIR(labels):
	IRLbl_VALUE = IRLbl(labels)
	return np.mean(IRLbl_VALUE)
def ML_ROS(all_labels, Preset_MeanIR_value=None, sample_size =32):
	N, C = all_labels.shape
	MeanIR_value = MeanIR(all_labels) if Preset_MeanIR_value is None else Preset_MeanIR_value
	IRLbl_value = IRLbl(all_labels)
	indices_per_class = {}
	minority_classes = []
	for i in range(C):
		ids = all_labels[:,i] == 1
		indices_per_class[i] = [ii for ii, x in enumerate(ids) if x ]
		if IRLbl_value[i] > MeanIR_value:
			minority_classes.append(i)
	new_all_labels = all_labels
	oversampled_ids = []
	for i in minority_classes:
		while True:
			pick_id = list(np.random.choice(indices_per_class[i], sample_size))
			indices_per_class[i].extend(pick_id)
			# recalculate the IRLbl_value
			new_all_labels = np.concatenate([new_all_labels, all_labels[pick_id]], axis=0)
			oversampled_ids.extend(pick_id)
			if IRLbl(new_all_labels)[i] <= MeanIR_value :
				break
			print("oversample length:{}".format(len(oversampled_ids)), end='\r')
			
	oversampled_ids = np.array(oversampled_ids)
	return new_all_labels

if __name__== '__main__':
	#data_file = read_all_image()
	data_file = merge_two_datasets()
	if args.vis:
		plot_n_samples_each_cate(data_file['Training_Set']['label'])
		plot_n_labels_each_instance(data_file['Training_Set']['label'])
	# perform resample and visualize
	resampled_labels = ML_ROS(data_file['Training_Set']['label'], Preset_MeanIR_value=2.0)
	plot_n_samples_each_cate(resampled_labels)
	pickle.dump(data_file, open(args.save_path, 'wb'))
	
