import collections

Dataset_Info = collections.namedtuple("Dataset_Info", 'data_file categories')
class PATH(object):
	def __init__(self, opt=None):
		self.Mixed_EXPR = Dataset_Info(data_file = '/media/Samsung/Aff-wild2-Challenge/exps/single_task/create_new_training_set_EXPR/create_annotation_file/mixed_EXPR_annotations.pkl',
			categories = ['Neutral','Anger','Disgust','Fear','Happiness','Sadness','Surprise'])
		self.Mixed_AU = Dataset_Info(data_file= '/media/Samsung/Aff-wild2-Challenge/exps/single_task/create_new_training_set_AU/create_annotation_file/mixed_AU_annotations.pkl',
			categories = ['AU1', 'AU2', 'AU4', 'AU6', 'AU12', 'AU15', 'AU20', 'AU25'])
		self.Mixed_VA = Dataset_Info(data_file= '/media/Samsung/Aff-wild2-Challenge/exps/single_task/create_new_training_set_VA/create_annotation_file/mixed_VA_annotations.pkl',
                        categories = ['valence', 'arousal'])
		self.Aff_wild2 = Dataset_Info(data_file = '/media/Samsung/Aff-wild2-Challenge/annotations/annotations.pkl',
			categories = {'AU': ['AU1', 'AU2', 'AU4', 'AU6', 'AU12', 'AU15', 'AU20', 'AU25'],
                            'EXPR':['Neutral','Anger','Disgust','Fear','Happiness','Sadness','Surprise'],
                            'VA':['valence', 'arousal']})
		# pytorch benchmark
		self.MODEL_DIR = '/media/Samsung/pytorch-benchmarks/models/'


