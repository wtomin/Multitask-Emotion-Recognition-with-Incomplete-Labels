from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.is_train = False
        self._parser.add_argument('--test_tasks_seq',default = ['AU', 'EXPR', 'VA'], type=str, nargs="+",
                                                        help='The sequence of tasks to be trained.')
        self._parser.add_argument('--save_path', type=str, default='output.pkl', help=' save of pickle path (predictions and possibly metric)')