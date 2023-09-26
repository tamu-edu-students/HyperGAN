import argparse
import os
from util import util


class BaseOptions():

    def __init__(self) -> None:
        self.initialized = False
        self.arg_options = None
        self.parser = None
    
    def initialize(self):

        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--model', type=str, default='cycle_gan', help='chooses which model to use. [cycle_gan | etc]')
        self.parser.add_argument('--epochs', type=int, default=200)
        self.parser.add_argument('--epoch_decay', type=int, default=100, help='epoch to start decaying learning rate')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
        self.parser.add_argument('--beta_1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--checkpoint_interval', type=int, default=10, help='interval between saving model checkpoints')
        self.parser.add_argument('--adversarial_loss_mode', default='lsgan', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
        self.parser.add_argument('--gradient_penalty_mode', default='none', choices=['none', 'dragan', 'wgan-gp'])
        self.parser.add_argument('--gradient_penalty_weight', type=float, default=10.0)
        self.parser.add_argument('--cycle_loss_weight', type=float, default=10.0)
        self.parser.add_argument('--identity_loss_weight', type=float, default=0.0)
        self.parser.add_argument('--pool_size', type=int, default=50)  # pool size to store fake samples

        self.initialized = True

    def gather_options(self):

        if not self.initialized:
            self.initialize()

        opt = self.parser.parse_known_args()
        model_name = opt.model

        #model_option_setter = models.get_option_setter(model_name)

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        #dataset_option_setter = data.get_option_setter(dataset_name)
        #parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        return self.parser.parse_args()
    
    
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

        # # set gpu ids
        # str_ids = opt.gpu_ids.split(',')
        # opt.gpu_ids = []
        # for str_id in str_ids:
        #     id = int(str_id)
        #     if id >= 0:
        #         opt.gpu_ids.append(id)
        # if len(opt.gpu_ids) > 0:
        #     torch.cuda.set_device(opt.gpu_ids[0])

        # self.opt = opt
        return self.opt
    