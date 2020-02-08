from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.is_train = False
        self._parser.add_argument('--teacher_model_path',default = '', type=str,  help='the model to be evaluated')
        self._parser.add_argument("--eval_with_teacher", action='store_true' )
        self._parser.add_argument('--mode', type=str, default='Validation', choices=['Validation', 'Test'], help='Whether \
        	evaluate it on the validation set or the test set.')
        self._parser.add_argument('--ensemble', action='store_true')
        self._parser.add_argument("--n_students", type=int, default=5)
        self._parser.add_argument("--save_dir", type=str, default = "Predictions")

        self.is_train = False

