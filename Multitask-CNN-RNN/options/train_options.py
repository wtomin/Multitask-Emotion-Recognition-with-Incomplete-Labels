from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self._parser.add_argument('--print_freq_s', type=int, default= 5, help='frequency of showing training results on console')
        self._parser.add_argument('--lr_policy', type=str, default = 'step', choices = ['lambda', 'step', 'plateau'])
        self._parser.add_argument('--lr_decay_epochs', type=int, default= 3, help="learning rate decays by 0.1 after every # epochs (lr_policy is 'step')")
        self._parser.add_argument('--teacher_nepochs', type=int, default= 3, help='# of epochs to train')
        self._parser.add_argument('--student_nepochs', type=int, default= 3, help='# of epochs for student to train')
        self._parser.add_argument('--n_students', type=int, default= 5, help='# of students')
        self._parser.add_argument('--pretrained_resnet50_model', type=str, default = '', help='pretrained model')
        self._parser.add_argument('--lr_F', type=float, default=0.0001, help='initial learning rate for G adam')
        self._parser.add_argument('--F_adam_b1', type=float, default=0.5, help='beta1 for G adam')
        self._parser.add_argument('--F_adam_b2', type=float, default=0.999, help='beta2 for G adam')
        self._parser.add_argument('--optimizer', type=str, default='Adam', choices = ['Adam', 'SGD'])
        self._parser.add_argument('--lambda_AU', type=float, default= 1., help='weight for AU.')
        self._parser.add_argument('--lambda_EXPR', type=float, default= 1., help='weight for EXPR.')
        self._parser.add_argument('--lambda_V', type=float, default= 1., help='weight for valence.')
        self._parser.add_argument('--lambda_A', type=float, default= 1., help='weight for arousal.')
        self._parser.add_argument('--lambda_ccc', type=float, default= 1., help='weight for ccc loss in (CE + lambda_ccc*ccc).')

        self.is_train = True
