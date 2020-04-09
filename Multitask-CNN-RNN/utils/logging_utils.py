import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
def draw_plots(df, title, save_path):
	plt.figure()
	for i, key in enumerate(df.keys()):
		if key !='loss':
			plt.plot(df.index, df[key], linewidth=2, label = key)
	plt.legend()
	plt.title(title)
	plt.xlabel("Iterations")
	plt.savefig(save_path, bbox_inches='tight')
	plt.clf()
	plt.cla()

	
def save_plots(data_dict, train_save_path, val_save_path):
	train_df = data_dict['training']
	val_df = data_dict['validation']
	draw_plots(train_df, 'Training Losses', train_save_path)
	draw_plots(val_df, 'Validation Metrics', val_save_path)
