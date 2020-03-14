import os
import pickle
from sklearn.metrics import f1_score
import  numpy as np
from tqdm import tqdm
import glob
# this script is used to evalute the pretrained models on the Aff-wild2 validation set
AU_list = ['AU1','AU2','AU4','AU6','AU12','AU15','AU20','AU25']
all_crop_aligned = '/media/Samsung/Aff-wild2-Challenge/cropped_aligned' # containing subdirectories with video name. Each subdirectory contains a sequence of crop-aligned faces
annotation_file = '/media/Samsung/Aff-wild2-Challenge/annotations/annotations.pkl' # annotation file created from create_annotation_file/Aff-wild2/create_train_val_annotation_file.py

input_model_dir = 'Multitask-CNN'
data = pickle.load(open(annotation_file, 'rb'))
save_val_results = 'save_val'
"""
Evalution Metrics: F1 score, accuracy and CCC
"""
def averaged_f1_score(input, target):
	N, label_size = input.shape
	f1s = []
	for i in range(label_size):
		f1 = f1_score(input[:, i], target[:, i])
		f1s.append(f1)
	return np.mean(f1s), f1s
def accuracy(input, target):
	assert len(input.shape) == 1
	return sum(input==target)/input.shape[0]
def averaged_accuracy(x, y):
	assert len(x.shape) == 2
	N, C =x.shape
	accs = []
	for i in range(C):
		acc = accuracy(x[:, i], y[:, i])
		accs.append(acc)
	return np.mean(accs), accs
def CCC_score(x, y):
	vx = x - np.mean(x)
	vy = y - np.mean(y)
	rho = np.sum(vx * vy) / (np.sqrt(np.sum(vx**2)) * np.sqrt(np.sum(vy**2)))
	x_m = np.mean(x)
	y_m = np.mean(y)
	x_s = np.std(x)
	y_s = np.std(y)
	ccc = 2*rho*x_s*y_s/(x_s**2 + y_s**2 + (x_m - y_m)**2)
	return ccc
def VA_metric(x, y):
	items = [CCC_score(x[:,0], y[:,0]), CCC_score(x[:,1], y[:,1])]
	return items, sum(items)
def EXPR_metric(x, y): 
	if not len(x.shape) == 1:
		if x.shape[1] == 1:
			x = x.reshape(-1)
		else:
			x = np.argmax(x, axis=-1)
	if not len(y.shape) == 1:
		if y.shape[1] == 1:
			y = y.reshape(-1)
		else:
			y = np.argmax(y, axis=-1)
	f1 = f1_score(x, y, average= 'macro')
	acc = accuracy(x, y)
	return [f1, acc], 0.67*f1 + 0.33*acc
def AU_metric(x, y):
	f1_av,_  = averaged_f1_score(x, y)
	x = x.reshape(-1)
	y = y.reshape(-1)
	acc_av  = accuracy(x, y)
	return [f1_av, acc_av], 0.5*f1_av + 0.5*acc_av
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
def predict_on_val_set(data, original_task):
	assert original_task in ['AU', 'VA', 'EXPR']
	data = data["{}_Set".format(original_task)]['Validation_Set']
	for video in data.keys():
		image_dir = os.path.join(all_crop_aligned, video)
		save_dir = os.path.join(save_val_results+"_{}_Set".format(original_task), video)
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
			order = 'python run_pretrained_model.py --image_dir {} --model_type CNN '.format(image_dir) +\
			'--batch_size 12 --eval_with_teacher --eval_with_students --save_dir {} '.format(save_dir) + \
			'--workers 8 --ensemble'
			os.system(order)
def evaluate_on_val_set(data, task):
	assert task in ['AU', 'VA', 'EXPR']
	data = data["{}_Set".format(task)]['Validation_Set']
	labels = []
	preds = {}
	read_functions = {'AU': read_AU, 'VA':read_VA, 'EXPR': read_Expr}
	eval_functions = {'AU': AU_metric, 'VA':VA_metric, 'EXPR': EXPR_metric}
	for video in tqdm(data.keys()):
		prediction_dir = os.path.join(save_val_results+"_{}_Set".format(task), video)
		model_list = ['teacher']
		model_list += ['student_{}'.format(i) for i in range(5)]
		model_list += ['merged']
		try:
			label = data[video]['label']
		except:
			try:
				label = data[video][AU_list].values
			except:
				label = data[video][['valence', 'arousal']].values
		labels.append(label)
		for model_name in model_list:
			txt_file = os.path.join(prediction_dir, model_name, '{}.txt'.format(task))
			assert os.path.exists(txt_file)
			pred = read_functions[task](txt_file)
			if pred.shape[0] != label.shape[0]:
				assert len(pred) > len(label)
				# this is because the 'run_pretrained_model.py' will predict every frame in the directory.
				# however, some frame has face, but does not have ground truth label (EXPR)
				label_frames_ids = data[video]['frames_ids'].values
				image_dir = os.path.join(all_crop_aligned, video)
				frames = sorted(glob.glob(os.path.join(image_dir, '*.jpg'))) # it depends on your extensions
				assert len(frames) == len(pred)
				frame_ids = [int(os.path.basename(x).split(".")[0]) for x in frames]
				mask = np.array([id-1 in label_frames_ids for id in frame_ids]) # the frame format is "00001.jpg"
				pred = pred[mask]
			if model_name not in preds.keys():
				preds[model_name] = []
			preds[model_name].append(pred)
	labels = np.concatenate(labels, axis=0)
	preds = dict([(key, np.concatenate(preds[key], axis=0)) for key in preds])
	for model_name in model_list:
		res = eval_functions[task](preds[model_name], labels)
		print("Model {} performance on {}: {} ({}, {})".format(model_name, task, res[1], res[0][0], res[0][1]))

if __name__=='__main__':
	print("For AU, the performance format is: Result (F1 score, Accuracy)")
	print("For EXPR, the performance format is: Result (F1 score, Accuracy)")
	print("For VA, the performance format is: Result (Valence CCC, Arousal CCC)")

	predict_on_val_set(data, 'AU')
	evaluate_on_val_set(data, 'AU')
	predict_on_val_set(data, 'VA')
	evaluate_on_val_set(data, 'VA')
	predict_on_val_set(data, 'EXPR')
	evaluate_on_val_set(data, 'EXPR')



