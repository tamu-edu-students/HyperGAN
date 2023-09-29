import itertools
from .base_model import BaseModel, BaseTrainer
from . import networks
import tensorflow as tf
import tensorflow.keras as keras
import pylib as py
import data_ops as dops
import functools


class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        self.set_input(opt)
        self.create_networks(opt)

    def create_networks(self, opt):
        
        if self.isTrain:
            self.model_names = ['G_A2B', 'G_B2A', 'D_A', 'D_B']
            self.G_A2B = networks.ResnetGenerator(input_shape=(opt.crop_size, opt.crop_size, 3))
            self.G_B2A = networks.ResnetGenerator(input_shape=(opt.crop_size, opt.crop_size, 3))
            self.D_A = networks.ConvDiscriminator(input_shape=(opt.crop_size, opt.crop_size, 3))
            self.D_B = networks.ConvDiscriminator(input_shape=(opt.crop_size, opt.crop_size, 3))

            self.d_loss_fn, self.g_loss_fn = networks.get_adversarial_losses_fn(opt.adversarial_loss_mode)
            self.cycle_loss_fn = tf.losses.MeanAbsoluteError()
            self.identity_loss_fn = tf.losses.MeanAbsoluteError()

            self.G_lr_scheduler = networks.LinearDecay(opt.lr, opt.epochs * self.len_dataset, opt.epoch_decay * self.len_dataset)
            self.D_lr_scheduler = networks.LinearDecay(opt.lr, opt.epochs * self.len_dataset, opt.epoch_decay * self.len_dataset)
            self.G_optimizer = keras.optimizers.Adam(learning_rate=self.G_lr_scheduler, beta_1=opt.beta_1)
            self.D_optimizer = keras.optimizers.Adam(learning_rate=self.D_lr_scheduler, beta_1=opt.beta_1)

        else:  # during test time, only load Gs
            self.model_names = ['G_A2B', 'G_B2A']
            self.G_A2B = networks.ResnetGenerator(input_shape=(opt.crop_size, opt.crop_size, 3))
            self.G_B2A = networks.ResnetGenerator(input_shape=(opt.crop_size, opt.crop_size, 3))
        
        print('networks created: ', self.model_names)

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser
    
    def set_input(self, opt):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.A_img_paths = py.glob(py.join(opt.datasets_dir, opt.dataroot, 'trainA'), '*.jpg')
        self.B_img_paths = py.glob(py.join(opt.datasets_dir, opt.dataroot, 'trainB'), '*.jpg')
        self.A_B_dataset, self.len_dataset = dops.make_zip_dataset(self.A_img_paths, self.B_img_paths, opt.batch_size, opt.load_size, opt.crop_size, training=True, repeat=False)

        self.A2B_pool = dops.ItemPool(opt.pool_size)
        self.B2A_pool = dops.ItemPool(opt.pool_size)

        self.A_img_paths_test = py.glob(py.join(opt.datasets_dir, opt.dataroot, 'testA'), '*.jpg')
        self.B_img_paths_test = py.glob(py.join(opt.datasets_dir, opt.dataroot, 'testB'), '*.jpg')
        self.A_B_dataset_test, _ = dops.make_zip_dataset(self.A_img_paths_test, self.B_img_paths_test, opt.batch_size, opt.load_size, opt.crop_size, training=False, repeat=True)

        self.output_dir = py.join('output', opt.dataroot)
        py.mkdir(self.output_dir)

    def forward(self, A, B):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        self.A2B = self.G_A2B(A, training=self.isTrain)
        self.B2A = self.G_B2A(B, training=self.isTrain)
        self.A2B2A = self.G_B2A(self.A2B, training=self.isTrain)
        self.B2A2B = self.G_A2B(self.B2A, training=self.isTrain)
        self.A2A = self.G_B2A(A, training=self.isTrain)
        self.B2B = self.G_A2B(B, training=self.isTrain)
    
    @tf.function
    def backward_G(self, A, B, opt):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        with tf.GradientTape() as t:

            self.A2B = self.G_A2B(A, training=self.isTrain)
            self.B2A = self.G_B2A(B, training=self.isTrain)
            self.A2B2A = self.G_B2A(self.A2B, training=self.isTrain)
            self.B2A2B = self.G_A2B(self.B2A, training=self.isTrain)
            self.A2A = self.G_B2A(A, training=self.isTrain)
            self.B2B = self.G_A2B(B, training=self.isTrain)

            A2B_d_logits = self.D_B(self.A2B, training=True)
            B2A_d_logits = self.D_A(self.B2A, training=True)

            A2B_g_loss = self.g_loss_fn(A2B_d_logits)
            B2A_g_loss = self.g_loss_fn(B2A_d_logits)
            A2B2A_cycle_loss = self.cycle_loss_fn(A, self.A2B2A)
            B2A2B_cycle_loss = self.cycle_loss_fn(B, self.B2A2B)
            A2A_id_loss = self.identity_loss_fn(A, self.A2A)
            B2B_id_loss = self.identity_loss_fn(B, self.B2B)

            G_loss = (A2B_g_loss + B2A_g_loss) + (A2B2A_cycle_loss + B2A2B_cycle_loss) * opt.cycle_loss_weight + (A2A_id_loss + B2B_id_loss) * opt.identity_loss_weight
        
        G_grad = t.gradient(G_loss, self.G_A2B.trainable_variables + self.G_B2A.trainable_variables)
        self.G_optimizer.apply_gradients(zip(G_grad, self.G_A2B.trainable_variables + self.G_B2A.trainable_variables))

        return {'A2B_g_loss': A2B_g_loss,
                      'B2A_g_loss': B2A_g_loss,
                      'A2B2A_cycle_loss': A2B2A_cycle_loss,
                      'B2A2B_cycle_loss': B2A2B_cycle_loss,
                      'A2A_id_loss': A2A_id_loss,
                      'B2B_id_loss': B2B_id_loss}
    @tf.function
    def backward_D(self, A, B, opt):
        with tf.GradientTape() as t:    
            A_d_logits = self.D_A(A, training=True)
            B2A_d_logits =  self.D_A(self.B2A, training=True)
            B_d_logits =  self.D_B(B, training=True)
            A2B_d_logits =  self.D_B(self.A2B, training=True)

            A_d_loss, B2A_d_loss =  self.d_loss_fn(A_d_logits, B2A_d_logits)
            B_d_loss, A2B_d_loss =  self.d_loss_fn(B_d_logits, A2B_d_logits)
            D_A_gp = networks.gradient_penalty(functools.partial(self.D_A, training=True), A, self.B2A, mode=opt.gradient_penalty_mode)
            D_B_gp = networks.gradient_penalty(functools.partial(self.D_B, training=True), B, self.A2B, mode=opt.gradient_penalty_mode)

            D_loss = (A_d_loss + B2A_d_loss) + (B_d_loss + A2B_d_loss) + (D_A_gp + D_B_gp) * opt.gradient_penalty_weight

        D_grad = t.gradient(D_loss, self.D_A.trainable_variables + self.D_B.trainable_variables)
        self.D_optimizer.apply_gradients(zip(D_grad, self.D_A.trainable_variables + self.D_B.trainable_variables))

        return {'A_d_loss': A_d_loss + B2A_d_loss,
            'B_d_loss': B_d_loss + A2B_d_loss,
            'D_A_gp': D_A_gp,
            'D_B_gp': D_B_gp}
    
    @tf.function
    def optimize_parameters(self, A, B, opt):
        """Update network weights; it will be called in every training iteration."""

        # self.forward(A, B)               # first call forward to calculate intermediate results
        
        G_loss_dict = self.backward_G(A, B, opt)
        
        self.A2B = self.A2B_pool(self.A2B_pool)
        self.B2A = self.B2A_pool(self.B2A_pool)

        D_loss_dict = self.backward_D(A, B, opt)

        return G_loss_dict, D_loss_dict
        
    @tf.function
    def sample(self, A, B):
        
        self.A2B = self.G_A2B(A, training=False)
        self.B2A = self.G_B2A(B, training=False)
        self.A2B2A = self.G_B2A(self.A2B, training=False)
        self.B2A2B = self.G_A2B(self.B2A, training=False)

        return self
    
class Trainer(BaseTrainer):

    def __init__():
        pass

    def forward(self, A, B):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        self.A2B = self.G_A2B(A, training=self.isTrain)
        self.B2A = self.G_B2A(B, training=self.isTrain)
        self.A2B2A = self.G_B2A(self.A2B, training=self.isTrain)
        self.B2A2B = self.G_A2B(self.B2A, training=self.isTrain)
        self.A2A = self.G_B2A(A, training=self.isTrain)
        self.B2B = self.G_A2B(B, training=self.isTrain) 
