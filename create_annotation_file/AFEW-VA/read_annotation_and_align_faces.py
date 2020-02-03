import os
import json
import math
import pandas
from PIL import Image
import numpy as np
import glob
import pickle
from tqdm import tqdm
from matplotlib import pyplot as plt
np.random.seed(0)

def crop_face(image, keypoints, rotate = True, quiet_mode=True):
    lex, ley = (keypoints[36] + keypoints[39])/2
    rex, rey = (keypoints[42] + keypoints[45])/2
    rmx, rmy = keypoints[54]
    lmx, lmy = keypoints[48]
    nex, ney = keypoints[33]
    # roation using PIL image

    if rotate:
        angle = calculate_angle(lex, ley, rex, rey)
        image, lex, ley, rex, rey, lmx, lmy, rmx, rmy \
            = image_rote(image, angle, lex, ley, rex, rey, lmx, lmy, rmx, rmy) 
    eye_width = rex - lex # distance between two eyes 
    ecx, ecy = (lex + rex) / 2.0, (ley + rey) / 2.0 # the center between two eyes 
    mouth_width = rmx - lmx  
    mcx, mcy = (lmx + rmx) / 2.0, (lmy + rmy) / 2.0 #mouth center coordinate
    em_height = mcy - ecy # height between mouth center to eyes center
    fcx, fcy = (ecx + mcx) / 2.0, (ecy + mcy) / 2.0 # face center
    # face 
    if eye_width > em_height: 
        alpha = eye_width 
    else: 
        alpha = em_height 
    g_beta = 2.0 
    g_left = fcx - alpha / 2.0 * g_beta 
    g_upper = fcy - alpha / 2.0 * g_beta 
    g_right = fcx + alpha / 2.0 * g_beta 
    g_lower = fcy + alpha / 2.0 * g_beta 
    g_face = image.crop((g_left, g_upper, g_right, g_lower)) 
    
    return g_face

def image_rote(img, angle, elx, ely, erx, ery, mlx, mly, mrx, mry, expand=1):
    w,h= img.size
    img = img.rotate(angle, expand=expand)  #whether to expand after rotation
    if expand == 0: 
        elx, ely = pos_transform_samesize(angle, elx, ely, w, h) 
        erx, ery = pos_transform_samesize(angle, erx, ery, w, h) 
        mlx, mly = pos_transform_samesize(angle, mlx, mly, w, h) 
        mrx, mry = pos_transform_samesize(angle, mrx, mry, w, h) 
    if expand == 1: 
        elx, ely = pos_transform_resize(angle, elx, ely, w, h) 
        erx, ery = pos_transform_resize(angle, erx, ery, w, h) 
        mlx, mly = pos_transform_resize(angle, mlx, mly, w, h) 
        mrx, mry = pos_transform_resize(angle, mrx, mry, w, h) 
    return img, elx, ely, erx, ery, mlx, mly, mrx, mry

def calculate_angle(elx, ely, erx, ery): 
    """
    calculate image rotate angle
    :param elx: lefy eye x
    :param ely: left eye y
    :param erx: right eye x
    :param ery: right eye y
    :return: rotate angle
    """ 
    dx = erx - elx 
    dy = ery - ely 
    angle = math.atan(dy / dx) * 180 / math.pi 
    return angle

def pos_transform_resize(angle, x, y, w, h): 
    """
    after rotation, new coordinate with expansion
    :param angle:
    :param x:
    :param y:
    :param w:
    :param h:
    :return:
    """ 
    angle = angle * math.pi / 180 
    matrix = [ math.cos(angle), math.sin(angle), 0.0, -math.sin(angle), math.cos(angle), 0.0 ] 
    def transform(x, y, matrix=matrix): 
        (a, b, c, d, e, f) = matrix 
        return a * x + b * y + c, d * x + e * y + f # calculate output size 
    xx = [] 
    yy = [] 
    for x_, y_ in ((0, 0), (w, 0), (w, h), (0, h)): 
        x_, y_ = transform(x_, y_) 
        xx.append(x_) 
        yy.append(y_) 
    ww = int(math.ceil(max(xx)) - math.floor(min(xx))) 
    hh = int(math.ceil(max(yy)) - math.floor(min(yy))) 
    # adjust center 
    cx, cy = transform(w / 2.0, h / 2.0) 
    matrix[2] = ww / 2.0 - cx 
    matrix[5] = hh / 2.0 - cy 
    tx, ty = transform(x, y) 
    return tx, ty

def pos_transform_samesize(angle, x, y, w, h): 
    """
    after rotation, new coordinate without expansion
    :param angle:
    :param x:
    :param y:
    :param w:
    :param h:
    :return:
    """ 
    angle = angle * math.pi / 180 
    matrix = [ math.cos(angle), math.sin(angle), 0.0, -math.sin(angle), math.cos(angle), 0.0 ] 
    def transform(x, y, matrix=matrix): 
        (a, b, c, d, e, f) = matrix 
        return a * x + b * y + c, d * x + e * y + f 
    cx, cy = transform(w / 2.0, h / 2.0) 
    matrix[2] = w / 2.0 - cx 
    matrix[5] = h / 2.0 - cy 
    x, y = transform(x, y) 
    return x, y
def PIL_image_convert(cv2_im):
    cv2_im = cv2.cvtColor(cv2_im,cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im)
    return pil_im
def create_annotation_file():
	root_dir= '.'
	output_dir = 'cropped_aligned_faces'
	data_file_path = 'annotations.pkl'
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	subfolders = ['{0:02d}'.format(i) for i in range(1, 13)]
	subfolders = [x for x in subfolders if os.path.isdir(os.path.join(root_dir, x))]
	all_videos_dirs = []
	for subfolder in subfolders:
		subsubfolders = os.listdir(os.path.join(root_dir, subfolder))
		all_videos_dirs.extend([os.path.join(root_dir, subfolder, x) for x in subsubfolders if os.path.isdir(os.path.join(root_dir, subfolder, x))])
	all_videos_names = ['_'.join(x.split('/')[-2:]) for x in all_videos_dirs]
	data_file = {}
	ids = np.random.permutation(len(all_videos_dirs))
	all_videos_names = [all_videos_names[i] for  i  in ids]
	all_videos_dirs = [all_videos_dirs[i] for  i in ids]
	N = int(len(all_videos_dirs)*0.8)
	data_file['Training_Set'] = {}
	data_file['Validation_Set'] = {}
	for ii in range(2):
		if ii==0:
			video_dirs = all_videos_dirs[:N]
			video_names = all_videos_names[:N]
			mode = 'Training_Set'
		else:
			video_dirs = all_videos_dirs[N:]
			video_names = all_videos_names[N:]
			mode = 'Validation_Set'
		print("Extract :{}".format(mode))
		for video_dir, video_name in tqdm(zip(video_dirs, video_names), total = len(video_dirs)):
			assert video_name == '_'.join(video_dir.split('/')[-2:])
			data_file[mode][video_name] = {}
			frames = sorted(glob.glob(os.path.join(video_dir, '*.png')))
			json_file = glob.glob(os.path.join(video_dir, '*.json'))[0]
			with open(json_file, 'r') as f:
				json_dict = json.load(f)
			frames_dict = json_dict['frames']
			len_frames = len(list(frames_dict.keys()))
			assert len_frames == len(frames), "Detected frames length is different from the labeled frames length"
			valences, arousals = [],  []
			paths = []
			for id, frame_path in zip(sorted(frames_dict.keys()), frames):
				assert id==os.path.basename(frame_path).split('.')[0]
				img = Image.open(frame_path).convert("RGB")
				ldm = np.array(frames_dict[id]['landmarks'])
				arousal = frames_dict[id]['arousal']
				valence = frames_dict[id]['valence']
				arousals.append(arousal)
				valences.append(valence)
				save_path = os.path.join(output_dir, video_name, os.path.basename(frame_path))
				paths.append(os.path.abspath(save_path))
				if not os.path.exists(save_path):
					crop_aligned_face = crop_face(img, ldm)
					if not os.path.exists(os.path.join(output_dir, video_name)):
						os.makedirs(os.path.join(output_dir, video_name))
					crop_aligned_face.save(save_path)
			valences, arousals = np.array(valences), np.array(arousals)
			data_file[mode][video_name]['valence'] = valences/10.
			data_file[mode][video_name]['arousal'] = arousals/10. # rescale to -1.0, 1.0
			data_file[mode][video_name]['path'] = paths
	pickle.dump(data_file, open(data_file_path, 'wb'))
if __name__=='__main__':
	data_file_path = 'annotations.pkl'
	if not os.path.exists(data_file_path):
		create_annotation_file()
	data_file = pickle.load(open(data_file_path, 'rb'))
	all_samples_arousal = np.concatenate([data_file['Training_Set'][x]['arousal'] for x in data_file['Training_Set'].keys()] + 
		                                 [data_file['Validation_Set'][x]['arousal'] for x in data_file['Validation_Set'].keys()], axis=0)

	all_samples_valence = np.concatenate([data_file['Training_Set'][x]['valence'] for x in data_file['Training_Set'].keys()] + 
		                                 [data_file['Validation_Set'][x]['valence'] for x in data_file['Validation_Set'].keys()], axis=0)
	plt.hist2d(all_samples_valence , all_samples_arousal, bins=(20, 20), cmap=plt.cm.jet)
	plt.xlabel("Valence")
	plt.ylabel('Arousal')
	plt.colorbar()
	plt.show()









