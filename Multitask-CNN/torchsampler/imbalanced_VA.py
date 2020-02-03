import torch
import torch.utils.data
import torchvision
from tqdm import tqdm
import numpy as np
import random

class ImbalancedDatasetSampler_VA(torch.utils.data.sampler.Sampler):
	"""Samples elements randomly from a given list of indices for imbalanced dataset
	Arguments:
		indices (list, optional): a list of indices
		num_samples (int, optional): number of samples to draw
		callback_get_label func: a callback-like function which takes two arguments - dataset and index
	"""

	def __init__(self, dataset, indices=None, num_samples=None):
				
		# if indices is not provided, 
		# all elements in the dataset will be considered
		self.indices = list(range(len(dataset))) \
			if indices is None else indices

		# if num_samples is not provided, 
		# draw `len(indices)` samples in each iteration
		self.num_samples = len(self.indices) \
			if num_samples is None else num_samples
		 
		all_labels = dataset._get_all_label()
		N, C = all_labels.shape
		assert C == 2
		hist, x_edges, y_edges = np.histogram2d(all_labels[:, 0], all_labels[:, 1], bins=[20, 20])
		x_bin_id = np.digitize( all_labels[:, 0], bins = x_edges) - 1
		y_bin_id = np.digitize( all_labels[:, 1], bins = y_edges) - 1
		# for value beyond the edges, the function returns len(digitize_num), but it needs to be replaced by len(edges)-1
		x_bin_id[x_bin_id==20] = 20-1
		y_bin_id[y_bin_id==20] = 20-1
		weights = []
		for x, y in zip(x_bin_id, y_bin_id):
			assert hist[x, y]!=0
			weights += [1 / hist[x, y]] 
		
		self.weights = torch.DoubleTensor(weights)

	def __iter__(self):
		return (self.indices[i] for i in torch.multinomial(
			self.weights, self.num_samples, replacement=True))

	def __len__(self):
		return self.num_samples

