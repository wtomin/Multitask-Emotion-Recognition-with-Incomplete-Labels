import torch
import torch.utils.data
import torchvision
from tqdm import tqdm
import numpy as np
import random
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

class ImbalancedDatasetSampler_ML(torch.utils.data.sampler.Sampler):
	"""Samples elements randomly from a given list of indices for imbalanced dataset
	Arguments:
		indices (list, optional): a list of indices
		num_samples (int, optional): number of samples to draw
		callback_get_label func: a callback-like function which takes two arguments - dataset and index
	"""

	def __init__(self, dataset, indices=None, num_samples=None, Preset_MeanIR_value= 2., 
		               max_clone_percentage=50, sample_size=32):
				
		# if indices is not provided, 
		# all elements in the dataset will be considered
		self.indices = list(range(len(dataset))) \
			if indices is None else indices

		# if num_samples is not provided, 
		# draw `len(indices)` samples in each iteration
		self.num_samples = len(self.indices) \
			if num_samples is None else num_samples
		 
		all_labels = dataset._get_all_label()
		MeanIR_value = MeanIR(all_labels) if Preset_MeanIR_value ==0 else Preset_MeanIR_value
		IRLbl_value = IRLbl(all_labels)
		N, C = all_labels.shape
		indices_per_class = {}
		minority_classes = []
		maxSamplesToClone = N / 100 * max_clone_percentage
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
				if IRLbl(new_all_labels)[i] <= MeanIR_value or len(oversampled_ids)>=maxSamplesToClone :
					break
				print("oversample length:{}".format(len(oversampled_ids)), end='\r')
			if len(oversampled_ids) >=maxSamplesToClone:
				break
		oversampled_ids = np.array(oversampled_ids)
		weights = np.array([1.0/len(self.indices)] * len(self.indices))
		unique, counts =  np.unique(oversampled_ids, return_counts=True)
		for i, n in zip(unique, counts):
			weights[i] = weights[i]*n
		self.weights = torch.DoubleTensor(weights)

	def __iter__(self):
		return (self.indices[i] for i in torch.multinomial(
			self.weights, self.num_samples, replacement=True))

	def __len__(self):
		return self.num_samples

