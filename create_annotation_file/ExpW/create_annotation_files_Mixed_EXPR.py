import pickle
import os
import numpy as np
import argparse
from matplotlib import pyplot as plt
import matplotlib
import glob
import pandas as pd
from PIL import Image
from tqdm import tqdm
parser = argparse.ArgumentParser(description='read two annotations files')
parser.add_argument('--aff_wild2_pkl', type=str, default = '/media/Samsung/Aff-wild2-Challenge/annotations/annotations.pkl')
parser.add_argument('--ExpW_pkl', type=str, default = '/media/Samsung/ExpW_dataset/annotations.pkl')
parser.add_argument('--save_path', type=str, default='/media/Samsung/Aff-wild2-Challenge/exps/create_new_training_set_EXPR/create_annotation_file/mixed_EXPR_annotations.pkl')
args = parser.parse_args()
Expr_list = ['Neutral','Anger','Disgust','Fear','Happiness','Sadness','Surprise']
def read_aff_wild2():
	total_data = pickle.load(open(args.aff_wild2_pkl, 'rb'))
	# training set
	data = total_data['EXPR_Set']['Training_Set']
	paths = []
	labels = []
	for video in data.keys():
		df = data[video]
		labels.append(df['label'].values.astype(np.float32))
		paths.append(df['path'].values)
	# undersample the neutral samples by 10
	# undersample the happy and sad samples by 20
	paths = np.concatenate(paths, axis=0)
	labels = np.concatenate(labels, axis=0)
	# neutral
	keep_10 = np.array([True if i%10==0 else False for i in range(len(labels))])
	to_drop = labels == 0
	to_drop = to_drop * (~keep_10)
	labels = labels[~to_drop]
	paths = paths[~to_drop]
	# happy
	keep_2 = np.array([True if i%2==0 else False for i in range(len(labels))])
	to_drop = labels == 4
	to_drop = to_drop * (~keep_2)
	labels = labels[~to_drop]
	paths = paths[~to_drop]
	# sadness
	keep_2 = np.array([True if i%2==0 else False for i in range(len(labels))])
	to_drop = labels == 5
	to_drop = to_drop * (~keep_2)
	labels = labels[~to_drop]
	paths = paths[~to_drop]
	data = {'label': labels, 'path': paths}
	# validation set
	val_data = total_data['EXPR_Set']['Validation_Set']
	paths = []
	labels = []
	for video in val_data.keys():
		df = val_data[video]
		labels.append(df['label'].values.astype(np.float32))
		paths.append(df['path'].values)
	# undersample the neutral samples by 10
	# undersample the happy and sad samples by 20
	paths = np.concatenate(paths, axis=0)
	labels = np.concatenate(labels, axis=0)
	val_data = {'label':labels, 'path':paths}
	return data, val_data
def merge_two_datasets():
	data_aff_wild2, data_aff_wild2_val = read_aff_wild2()
	data_ExpW = pickle.load(open(args.ExpW_pkl, 'rb'))
	# change the label integer, because of the different labelling in two datasets
	ExpW_to_aff_wild2 = [1, 2, 3, 4, 5, 6, 0]
	data_ExpW['label'] = np.array([ExpW_to_aff_wild2[x] for x in data_ExpW['label']])
	data_merged = {'label': np.concatenate((data_aff_wild2['label'], data_ExpW['label']), axis=0),
	                'path': list(data_aff_wild2['path']) +data_ExpW['path']}
	print("Dataset\t"+"\t".join(Expr_list))
	print("Aff_wild2 dataset:\t" +"\t".join([str(sum(data_aff_wild2['label']==i)) for i in range(len(Expr_list))]))
	print("ExpW dataset:\t" +"\t".join([str(sum(data_ExpW['label']==i)) for i in range(len(Expr_list))]))
	return {'Training_Set': data_merged, 'Validation_Set': data_aff_wild2_val}

def plot_distribution(data):
	all_samples = data['label']
	histogram = np.zeros(len(Expr_list))
	for i in range(len(Expr_list)):
		find_true = sum(all_samples==i)
		histogram[i] =find_true/all_samples.shape[0]
	plt.bar(np.arange(len(Expr_list)), histogram)
	plt.xticks(np.arange(len(Expr_list)), Expr_list)
	plt.show()
if __name__== '__main__':
	#data_file = read_all_image()
	data_file = merge_two_datasets()
	pickle.dump(data_file, open(args.save_path, 'wb'))
	plot_distribution(data_file['Training_Set'])