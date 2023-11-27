import argparse
import os
from util import util
import models

class BaseOptions():

    def __init__(self) -> None:
        self.initialized = False
        self.arg_options = None
        
    
    def initialize(self, parser):

        parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--datasets_dir', default='datasets')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--model', type=str, default='cycle_gan', help='chooses which model to use. [cycle_gan | etc]')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--cuda', action='store_true', help='use GPU computation')
        parser.add_argument('--epochs', type=int, default=1000)
        parser.add_argument('--epoch_decay', type=int, default=100, help='epoch to start decaying learning rate')
        parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
        parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
        parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
        parser.add_argument('--beta_1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
        parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='models are saved here')
        parser.add_argument('--checkpoint_interval', type=int, default=10, help='interval between saving model checkpoints')
        parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
        parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
        parser.add_argument('--batch_size', type=int, default=1)
        parser.add_argument('--adversarial_loss_mode', default='lsgan', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
        parser.add_argument('--gradient_penalty_mode', default='none', choices=['none', 'dragan', 'wgan-gp'])
        parser.add_argument('--gradient_penalty_weight', type=float, default=10.0)
        parser.add_argument('--cycle_loss_weight', type=float, default=10.0)
        parser.add_argument('--identity_loss_weight', type=float, default=0.0)
        parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
        parser.add_argument('--pool_size', type=int, default=50)  # pool size to store fake samples
        parser.add_argument('--restore', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')

        self.initialized = True
        return parser

    def gather_options(self):

        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        opt, _ = parser.parse_known_args()
        

        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        # modify dataset-related parser options
       
        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    
    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt"""
        
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
    
    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        self.opt = opt
        return self.opt
    