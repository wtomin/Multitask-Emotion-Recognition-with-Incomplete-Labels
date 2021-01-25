import os
class Video_Processor(object):
    def __init__(self, size=112, nomask=True, grey=False, quiet=True,
                 tracked_vid=False, noface_save=False,
                 OpenFace_exe = 'OpenFace/build/bin/FeatureExtraction'):
        ''' Video Processor using OpenFace to do face detection and face alignment
        Given an input video, this processor will create a directory where all cropped and aligned 
        faces are saved.

        Parameters: 
            size: int, default 112
                The output faces will be saved as images where the width and height are size pixels.
            nomask: bool, default True
                If True, the output face image will not be masked (mask the region except the face).
                Otherwise, the output face image will be masked, not containing background.
            grey: bool, default False
                If True, the output face image will be saved as greyscale images instead of RGB images.
            quiet: bool, default False
                If False, will print out the processing steps live.
            tracked_vid: bool, default False
                If True, will save the tracked video, which is an output video with detected landmarks.
            noface_save: bool, default False
                If True, those frames where face detection is failed will be saved (blank image); 
                else those failed frames will not saved.
            OpenFace_exe: String, default is 'OpenFace/build/bin/FeatureExtraction'
                By default, the OpenFace library is installed in the same directory as Video_Processor.
                It can be changed to the current OpenFace executable file.
        '''
        self.size = size
        self.nomask = nomask
        self.grey = grey
        self.quiet = quiet
        self.tracked_vid = tracked_vid
        self.noface_save = noface_save
        self.OpenFace_exe = OpenFace_exe 
        if not isinstance(self.OpenFace_exe, str) or not os.path.exists(self.OpenFace_exe):
            raise ValueError("OpenFace_exe has to be string object and needs to exist.")
        self.OpenFace_exe = os.path.abspath(self.OpenFace_exe)
    def process(self, input_video, output_dir=None):
        '''        
        Arguments:
            input_video: String
               The input video path, or the input sequence directory, where each image representing a frame, e.g. 001.jpg, 002.jpg, 003.jpg ... 200.jpg
            output_dir: String, default None
               The output faces will be saved in output_dir. By default the output_dir will be in 
               the same parent directory as the input video is.
        '''

        if not isinstance(input_video, str) or not os.path.exists(input_video):
            raise ValueError("input video has to be string object and needs to exist.")
        if os.path.isdir(input_video):
            assert len(os.listdir(input_video))>0, "Input sequence directory {} cannot be empty".format(input_video)
            arg_input = '-fdir'
        else:
            arg_input = '-f'

        input_video = os.path.abspath(input_video)
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(input_video), 
                os.path.basename(input_video).split('.')[0])
        if isinstance(output_dir, str):
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            else:
            	print("output dir exists: {}. Video processing skipped.".format(output_dir))
            	return
        else:
            raise ValueError("output_dir should be string object.")
        opface_option = " {} ".format(arg_input)+input_video + " -out_dir "+ output_dir +" -simsize "+ str(self.size)
        opface_option += " -2Dfp -3Dfp -pdmparams -pose -aus -gaze -simalign "

        if not self.noface_save:
            opface_option +=" -nobadaligned "
        if self.tracked_vid:
            opface_option +=" -tracked "
        if self.nomask:
            opface_option+= " -nomask"
        if self.grey:
            opface_option += " -g"
        if self.quiet:
            opface_option += " -q"
        # execution
        call = self.OpenFace_exe + opface_option
        os.system(call)



