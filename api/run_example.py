from Emotion_API import Emotion_API
import os
import torch
import time
example_video = 'examples/utterance_1.mp4'

# use non-temporal model
api = Emotion_API(
	device = torch.device('cpu'), # If GPU is available, change it to torch.device('cuda:0')
	use_temporal=False, # When using temporal model, the computation cost is larger than using non-temporal model.
	num_students = 1, # One can choose between the integers from 1 to 5. Larger number leads to larger computation cost.
	OpenFace_exe = './OpenFace/build/bin/FeatureExtraction', # if OpenFace is installed somewhere else, replace the path here
	length = 32, # This length applies when use_temporal=True. 
	#If the video length is smaller than 32, you can just leave it like this. 
	#The image sampler will sample some frames repeatedly to meet the length requirement, which can take longer.
	#If the video length is smaller than 32 and you want to save time, you can change the length to the video length.
	batch_size = 24, # The input tensor is (Batch_size, 3, 112, 112) for non-temporal model and (Batch_size, length, 3, 112, 112) for temporal model.
        )
api.run(example_video, csv_output='examples/multitask_preds.csv')
