import os
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from PIL import Image
import matplotlib.colors as mcolors
matplotlib.use('TkAgg')
font = {'family' : 'normal',
		'size'   : 24}
matplotlib.rc('font', **font)
CATEGORIES = {'AU': ['AU1', 'AU2', 'AU4', 'AU6', 'AU12', 'AU15', 'AU20', 'AU25'],
                            'EXPR':['Neutral','Anger','Disgust','Fear','Happiness','Sadness','Surprise'],
                            'VA':['valence', 'arousal']}
Best_AU_Thresholds = {'CNN': [0.1448537, 0.03918985, 0.13766725, 0.02652811, 0.40589422, 0.15572545,0.04808964, 0.10848708],
                      'CNN-RNN': {32: [0.4253935, 0.02641966, 0.1119782, 0.02978198, 0.17256933, 0.06369855, 0.07433069, 0.13828614],
                                  16: [0.30485213, 0.09509478, 0.59577084, 0.4417419, 0.4396544, 0.0452404,0.05204154, 0.0633798 ],
                                  8: [0.4365209 ,0.10177602, 0.2649502,  0.22586018, 0.3772219,  0.07532539, 0.07667687, 0.04306327]}}
color = mcolors.TABLEAU_COLORS['tab:blue']
AU_png_dir = 'AU_pngs'
def parse_txt(txt_file):
	with open(txt_file, 'r') as f:
		lines=f.readlines()
		lines = [l.strip() for l in lines]
		lines = lines[1:]
		lines = [l.split(',') for l in lines]
		lines = [[float(d) for d in l] for l in lines]
	return np.array(lines)
class Emotion_API(object):
	# given a inptu video file, and a directory containing all cropped faces
	# and a keypoints.csv, and a directory containing all predictions from five student models
	def __init__(self, video_file, root_dir, pred_dir,
		save_dir):
		self.video_file = video_file
		self.root_dir = root_dir
		self.pred_dir = pred_dir
		self.save_dir = save_dir

		self.video = cv2.VideoCapture(self.video_file)
		self.fps = int(round(self.video.get(cv2.CAP_PROP_FPS)))
		self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
		self.read_all()
	def read_all(self):
		au_txt = os.path.join(self.pred_dir, 'merged_raw', 'AU.txt')
		expr_txt = os.path.join(self.pred_dir, 'merged_raw', 'EXPR.txt')
		va_txt = os.path.join(self.pred_dir, 'merged_raw', 'VA.txt')
		data = {}
		data['AU'] = parse_txt(au_txt)
		data['EXPR'] = parse_txt(expr_txt)
		data['VA'] = parse_txt(va_txt)
		kps_df = os.path.join(self.root_dir, 'keypoints.csv')
		kps_df = pd.read_csv(kps_df)
		assert int(max(kps_df['frame'])) == self.total_frames
		assert int(sum(kps_df['detect'].values)) == data['AU'].shape[0]
		data['df'] = kps_df
		detects = kps_df['detect'].values
		original_id_2_detected_id = []
		prev_i = -1
		for d in detects:
			if d==1.0:
				prev_i +=1
			original_id_2_detected_id.append(max(0, prev_i))
		assert max(original_id_2_detected_id) == data['AU'].shape[0] - 1
		data['id2id'] = np.array(original_id_2_detected_id)
		self.data = data
		return data
	def make_video(self):
		# create four folders: frames, AUs, EXPRs, VAs
		frames_dir = os.path.join(self.save_dir, 'frames')
		AUs_dir = os.path.join(self.save_dir, 'AUs')
		EXPRs_dir = os.path.join(self.save_dir, 'EXPRs')
		VAs_dir = os.path.join(self.save_dir, 'VAs')
		for dir_path, make_func in zip([VAs_dir, AUs_dir, EXPRs_dir], [self.make_VA, self.make_AU, self.make_EXPR]):
			if not os.path.exists(dir_path):
				os.makedirs(dir_path)
				make_func(display=False, save_dir=dir_path)
		if not os.path.exists(frames_dir):
			os.makedirs(frames_dir)
			#create frames
			des = os.path.join(frames_dir, '%06d.png')
			cmd = 'ffmpeg -i '+self.video_file+' -filter:v fps=fps='+str(self.fps)+' '+des
			os.system(cmd)
			assert len(list(os.listdir(frames_dir))) >= self.total_frames
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		out_video_path = os.path.join(os.path.dirname(self.video_file), 'video.avi')
		out_video = cv2.VideoWriter(filename=out_video_path, 
			fourcc=fourcc, 
			frameSize=(1920, 1080),
			fps=self.fps)
		df = self.data['df']
		for i in tqdm(range(self.total_frames), total = self.total_frames):
			f = cv2.imread(os.path.join(frames_dir, '{:06}.png'.format(i+1)))
			kpts = df[df['frame']==(i+1)]
			if kpts['detect'].values[0] == 1.0:
				f = self.plot_rectange(f, kpts)
				prev_kpts = kpts
			else:
				f = self.plot_rectange(f, prev_kpts)
			au = cv2.imread(os.path.join(AUs_dir, '{:06}.png'.format(i)))
			expr = cv2.imread(os.path.join(EXPRs_dir, '{:06}.png'.format(i)))
			va = cv2.imread(os.path.join(VAs_dir, '{:06}.png'.format(i)))
			# f size limit (960, 540)
			new_frame = np.zeros((1080, 1920, 3)).astype(np.uint8)
			f = self.pad_image_to_size(f, size=(960, 740))
			new_frame[0:740, 0:960] = f 
			au = self.pad_image_to_size(au, size=(480, 740))
			new_frame[0:740, 960:960+480] = au
			expr = self.pad_image_to_size(expr, size=(480, 740))
			new_frame[0:740, 960+480:1920] = expr
			va = self.pad_image_to_size(va, size=(1920, 340))
			new_frame[740: 1080, :] = va
			out_video.write(new_frame)
		cv2.destroyAllWindows()
		out_video.release()

		# split audio from video_file
		audio_des = os.path.join(self.save_dir, 'output_audio.mp3')
		cmd = 'ffmpeg -i {} -q:a 0 -map a {}'.format(self.video_file, audio_des)
		os.system(cmd)
		new_video = os.path.join(self.save_dir, 'audiovideo.avi')
		cmd = 'ffmpeg -i {} -i {} -map 0:v -map 1:a -c:v copy -shortest {}'.format(
			out_video_path,
			audio_des,
			new_video)
		os.system(cmd)

	def decode_string(self, string):
		x = string[1:-1].split(',')
		x = [int(s) for s in x]
		return x
	def plot_rectange(self, image, keypoints):
	    lex, ley = self.decode_string(keypoints['left_eye'].values[0])
	    rex, rey = self.decode_string(keypoints['right_eye'].values[0])
	    rmx, rmy = self.decode_string(keypoints['mouth_right'].values[0])
	    lmx, lmy = self.decode_string(keypoints['mouth_left'].values[0])
	    nex, ney = self.decode_string(keypoints['nose'].values[0])
	    eye_width = rex - lex # distance between two eyes 
	    ecx, ecy = (lex + rex) / 2.0, (ley + rey) / 2.0 # the center between two eyes 
	    mouth_width = rmx - lmx  
	    mcx, mcy = (lmx + rmx) / 2.0, (lmy + rmy) / 2.0 #mouth center coordinate
	    em_height = mcy - ecy # height between mouth center to eyes center
	    fcx, fcy = (ecx + mcx) / 2.0, (ecy + mcy) / 2.0 # face center
	    g_beta = 2.0 
	    # face 
	    if eye_width > em_height: 
	        alpha = eye_width 
	    else: 
	        alpha = em_height 
	    g_left = int(round(fcx - alpha / 2.0 * g_beta ))
	    g_upper = int(round(fcy - alpha / 2.0 * g_beta)) 
	    g_right = int(round(fcx + alpha / 2.0 * g_beta)) 
	    g_lower = int(round(fcy + alpha / 2.0 * g_beta ))
	    cv2.rectangle(image, (g_left, g_upper), (g_right, g_lower), (0, 255, 0), 4)
	    return image
	def pad_image_to_size(self, image, size, pad_pixel=2):
		rw, rh = size
		new_image = np.ones((rh, rw, 3)).astype(np.uint8)*255
		h, w, _ = image.shape
		if (h/w) > ((rh - pad_pixel)/(rw-pad_pixel)):
			image = cv2.resize(image, (int((rh-pad_pixel)*w/h), rh-pad_pixel))
		else:
			image = cv2.resize(image, (rw - pad_pixel, int((rw-pad_pixel)*h/w)))
		h, w, _ = image.shape
		w_pad, h_pad = (rw - w)//2, (rh - h)//2
		top = h_pad
		bottom = h_pad + h
		left = w_pad
		right = w_pad + w
		new_image[top:bottom, left:right] = image
		return new_image

	def make_VA(self, display=False, save_dir=None):
		pred = self.data['VA'][self.data['id2id'], :]
		#pred: array of size (N, 2)
		assert pred.shape[0] == self.total_frames
		#min_v, max_v = pred.min(), pred.max()
		fig = plt.figure(1, figsize=(45, 9))
		ax = fig.add_subplot(111)
		ax.set_title("Valence & Arousal")
		ax.set_ylim([-1, 1])
		show_freq = 16
		cur_frame = 0 
		ax.set_xlim([0, show_freq])
		if display:
			fig.show()
			fig.canvas.draw()
		if save_dir is not None:
			assert os.path.exists(save_dir), 'save_dir does not exists:{}'.format(save_dir)
		while cur_frame < self.total_frames:
			p = pred[cur_frame]
			if cur_frame% (show_freq*self.fps) == 0:
				frames_ids = [cur_frame]
				arousals = [p[1]]
				valences = [p[0]]
			else:
				frames_ids.append(cur_frame)
				arousals.append(p[1])
				valences.append(p[0])
			ax.plot([i/self.fps for i in frames_ids], arousals,
				color = 'r', label='arousal')
			ax.plot([i/self.fps for i in frames_ids], valences,
				color = 'b', label='valence')
			ax.set_xlim([min([i/self.fps for i in frames_ids]), 
				min([i/self.fps for i in frames_ids])+show_freq])
			ax.set_ylim([-1, 1])
			ax.legend(loc='upper right')
			if display:
				fig.canvas.draw()
				fig.canvas.flush_events()
				time.sleep(0.1)
			if save_dir is not None:
				des = os.path.join(save_dir, '{:06}.png'.format(cur_frame))
				plt.savefig(des, dpi=60,
					bbox_inches='tight')
			ax.clear()
			ax.set_ylim([-1, 1])
			ax.set_title("Valence & Arousal")
			ax.set_xlabel("Seconds")
			cur_frame += 1
			print("{}/{}".format(cur_frame, self.total_frames), end='\r')

	def make_EXPR(self, display=False, save_dir=None):
		pred = self.data['EXPR'][self.data['id2id'], :]
		label_list = CATEGORIES['EXPR']
		assert pred.shape[0] == self.total_frames
		fig = plt.figure(2, figsize=(8, 9))
		ax = fig.add_subplot(111)
		pos = np.arange(len(label_list))
		im = ax.barh(pos, [0]*len(label_list), 
			align='center',
			height = 0.5,
			tick_label = label_list)
		ax.set_xlim([0, 1])
		ax.set_title("Expressions")
		if display:
			fig.show()
			fig.canvas.draw()
		if save_dir is not None:
			assert os.path.exists(save_dir), 'save_dir does not exists:{}'.format(save_dir)
		for i, p in tqdm(enumerate(pred), total= len(pred)):
			ax.barh(pos, p, 
				align='center',
				height = 0.5,
				tick_label = label_list,
				color=color)
			if display:
				fig.canvas.draw()
				fig.canvas.flush_events()
				time.sleep(0.1)
			if save_dir is not None:
				des = os.path.join(save_dir, '{:06}.png'.format(i))
				plt.savefig(des, dpi=60,
					bbox_inches='tight')
			ax.clear()
			ax.set_xlim([0, 1])
			ax.set_title("Expressions")

	def make_AU(self, display=False, save_dir = None):
		ts = np.array(Best_AU_Thresholds['CNN-RNN'][16])
		ns = 0.5/ts
		pred = self.data['AU'][self.data['id2id'], :]
		assert pred.shape[0] == self.total_frames
		label_list = CATEGORIES['AU']
		au_pngs = [os.path.join(AU_png_dir, au+'.png') for au in label_list]
		au_pngs = [Image.open(path) for path in au_pngs]
		fig = plt.figure(3, figsize=(8, 12))
		ax = fig.add_subplot(111)
		pos = np.arange(len(label_list))
		im = ax.barh(pos, [0]*len(label_list), 
			align='center',
			height = 0.8,
			#tick_label = [],
			color=color)
		ax.set_xlim([0, 1])
		ax.set_title("Facial Action Units")
		ax.yaxis.set_ticklabels([])
		yticks = ax.get_yticks()
		ax_news= []
		for i, (tick_pos, au_png) in enumerate(zip(yticks, au_pngs)):
			if i<=1:
				yloc= 0.11*(i+1)
			elif i in [4, 5, 6, 7]:
				yloc= 0.095*(i+1)
			else:
				yloc = 0.1*(i+1)
			ax_new = fig.add_axes([0.0, yloc, 0.115, 0.115])
			ax_new.imshow(np.array(au_png), cmap='gray', vmin=0, vmax=255)
			ax_new.get_xaxis().set_visible(False)
			ax_new.get_yaxis().set_visible(False)
			#ax_new.set_axis_off()
			ax_news.append(ax_new)

		if display:
			fig.show()
			fig.canvas.draw()

		if save_dir is not None:
			assert os.path.exists(save_dir), 'save_dir does not exists:{}'.format(save_dir)
		for i, p in tqdm(enumerate(pred), total=len(pred)):
			p = np.array(p)#*ns
			ax.barh(pos, p, 
				align='center',
				height = 0.8,
				#tick_label = [],
				color=color)
			if display:
				fig.canvas.draw()
				fig.canvas.flush_events()
				time.sleep(0.1)
			if save_dir is not None:
				des = os.path.join(save_dir, '{:06}.png'.format(i))
				plt.savefig(des, dpi=60,
					bbox_inches='tight')
			ax.clear()
			ax.set_xlim([0, 1])
			ax.set_title("Facial Action Units")
			ax.yaxis.set_ticklabels([])
			for ax_new in ax_news:
				ax_new.clear()
			ax_news = []
			for i, (tick_pos, au_png) in enumerate(zip(yticks, au_pngs)):
				if i<=1:
					yloc= 0.11*(i+1)
				elif i in [4, 5, 6, 7]:
					yloc= 0.095*(i+1)
				else:
					yloc = 0.1*(i+1)
				ax_new = fig.add_axes([0.0, yloc, 0.115, 0.115])
				ax_new.imshow(np.array(au_png), cmap='gray', vmin=0, vmax=255)
				ax_new.get_xaxis().set_visible(False)
				ax_new.get_yaxis().set_visible(False)
				#ax_new.set_axis_off()
				ax_news.append(ax_new)


if __name__ == '__main__':
	video_file = 'video84/video84.mp4'
	name = video_file.split('/')[-1].split('.')[0]
	save_dir = os.path.dirname(video_file)
	root_dir = os.path.join(save_dir, name)
	pred_dir = os.path.join(save_dir, 'preds')
	cmd = 'python MTCNN_alignment_with_video.py -i {} -o {} --size 112 -q'.format(save_dir, save_dir)
	# save cropped faces
	os.system(cmd)
	cmd = 'python run_pretrained_model.py --image_dir {} --model_type CNN-RNN --seq_len 32 --batch_size 6 --eval_with_students --save_dir {} --workers 4 --ensemble'.format(root_dir,
		pred_dir)
	# save predictions
	os.system(cmd)
	# generate images
	api = Emotion_API(video_file, root_dir, pred_dir, save_dir)
	api.make_video()




		

	
