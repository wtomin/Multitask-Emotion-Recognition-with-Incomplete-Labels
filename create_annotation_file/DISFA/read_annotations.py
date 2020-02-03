import os
import pickle
import numpy as np
import argparse
from matplotlib import pyplot as plt
import matplotlib
import glob
np.random.seed(0)
parser = argparse.ArgumentParser(description='save annotations')
parser.add_argument('--vis', action = 'store_true', 
                    help='whether to visualize the distribution')
parser.add_argument('--annot_dir', type=str, default = 'ActionUnit_Labels',
                    help='annotation dir')
parser.add_argument("--image_dir", type=str, default= '/media/Samsung/DISFA/Videos_LeftCamera_OpenFace_Aligned')

args = parser.parse_args()
def read_au(txt_prefix):
	au_list = [1, 2, 4, 6, 12, 15, 20, 25]
	aus = []
	for i in au_list:
		txt_file = txt_prefix+"{}.txt".format(i)
		with open(txt_file, 'r') as f:
			lines = f.readlines()
		lines = [x.strip() for x in lines]
		lines = [int(x.split(',')[1]) for x in lines]
		lines = [0 if x<2 else 1 for x in lines ] # if intensity is equal to or greater than 2, it is positive sample
		aus.append(lines)
	aus = np.stack(aus, axis=1)
	return aus

def plot_pie(AU_list, pos_freq, neg_freq):
	ploting_labels = [x+'+ {0:.2f}'.format(y) for x, y in zip(AU_list, pos_freq)] + [x+'- {0:.2f}'.format(y) for x, y in zip(AU_list, neg_freq)] 
	cmap = matplotlib.cm.get_cmap('coolwarm')
	colors = [cmap(x) for x in pos_freq] + [cmap(x) for x in neg_freq]
	fracs = np.ones(len(AU_list)*2)
	plt.pie(fracs, labels=ploting_labels, autopct=None, shadow=False, colors=colors,startangle =78.75)
	plt.title("AUs distribution")
	plt.show()
AU_list = ['AU1','AU2','AU4','AU6','AU12','AU15','AU20','AU25']
annot_dir = args.annot_dir
data_file = {}
videos = sorted(os.listdir(annot_dir))
# in total 27 videos
ids = np.random.permutation(len(videos))
videos = [videos[i] for i in ids]
train_videos = videos[:21]
val_videos = videos[21:]
data_file['Training_Set'] = {}
data_file['Validation_Set'] = {}
for video in train_videos:
	aus = read_au(annot_dir+'/{}/{}_au'.format(video, video))
	frames = sorted(glob.glob(os.path.join(args.image_dir, "LeftVideo"+video+"_comp", "LeftVideo"+video+"_comp_aligned", '*.bmp')))
	frames_id = [int(x.split("/")[-1].split(".")[0].split("_")[-1]) -1 for x in frames]
	assert len(aus)>=len(frames)
	frames = [frames[id] for id in frames_id]
	aus = aus[frames_id]
	data_file['Training_Set'][video] = {'label':aus, 'path':frames}
for video in val_videos:
	aus = read_au(annot_dir+'/{}/{}_au'.format(video, video))
	frames = sorted(glob.glob(os.path.join(args.image_dir, "LeftVideo"+video+"_comp", "LeftVideo"+video+"_comp_aligned", '*.bmp')))
	frames_id = [int(x.split("/")[-1].split(".")[0].split("_")[-1]) -1 for x in frames]
	assert len(aus)>= len(frames)
	frames = [frames[id] for id in frames_id]
	aus = aus[frames_id]
	data_file['Validation_Set'][video] = {'label':aus, 'path':frames}

if args.vis:
	total_dict = {**data_file['Training_Set'], **data_file['Validation_Set']}
	all_samples = np.concatenate([total_dict[x]['label'] for x in total_dict.keys()], axis=0)
	pos_freq = np.sum(all_samples, axis=0)/all_samples.shape[0]
	neg_freq = -np.sum(all_samples-1, axis=0)/all_samples.shape[0]
	print("pos_weight:", neg_freq/pos_freq)
	plot_pie(AU_list, pos_freq, neg_freq)
save_path = 'annotations.pkl'
pickle.dump(data_file, open(save_path, 'wb'))

