import itertools
from .base_model import BaseModel
from . import networks
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset.maskshadow_dataset import MaskImageDataset
import pylib as py
import functools
from PIL import Image
from util.util import LambdaLR, weights_init_normal, ReplayBuffer, QueueMask, mask_generator


class MaskShadowGAN(BaseModel):
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
        self.loss_names = ['D_A', 'G_A2B', 'cycle_ABA', 'idt_A', 'D_B', 'G_B2A', 'cycle_BAB', 'idt_B']
        self.create_networks(opt)


    def create_networks(self, opt):
        
        if self.isTrain:
            self.model_names = ['G_A2B', 'G_B2A', 'D_A', 'D_B']
            self.G_A2B = networks.Generator(opt.input_nc, opt.output_nc)
            self.G_B2A = networks.Generator_F2S(opt.output_nc, opt.input_nc)
            self.D_A = networks.Discriminator(opt.input_nc)
            self.D_B = networks.Discriminator(opt.output_nc)

            if opt.cuda:
                print("Using GPU")
                self.G_A2B.cuda()
                self.G_B2A.cuda()
                self.D_A.cuda()
                self.D_B.cuda()
            
            self.G_A2B.apply(weights_init_normal)
            self.G_B2A.apply(weights_init_normal)
            self.D_A.apply(weights_init_normal)
            self.D_B.apply(weights_init_normal)

            
            self.criterionGAN = torch.nn.MSELoss()  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdentity = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.G_A2B.parameters(), self.G_B2A.parameters()), lr=opt.lr, betas=(opt.beta_1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.D_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.D_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))
            self.optimizers.append('optimizer_G')
            self.optimizers.append('optimizer_D_A')
            self.optimizers.append('optimizer_D_B')
            self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G,
												   lr_lambda=LambdaLR(opt.epochs, opt.epoch_count, opt.epoch_decay).step)
            self.lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D_A,
													 lr_lambda=LambdaLR(opt.epochs, opt.epoch_count, opt.epoch_decay).step)
            self.lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D_B,
													 lr_lambda=LambdaLR(opt.epochs, opt.epoch_count, opt.epoch_decay).step)
            self.lrs.append('lr_scheduler_G')
            self.lrs.append('lr_scheduler_D_A')
            self.lrs.append('lr_scheduler_D_B')
            
        else:  # during test time, only load Gs
            self.model_names = ['G_A2B', 'G_B2A']
            self.G_A2B = networks.Generator(opt.input_nc, opt.output_nc)
            self.G_B2A = networks.Generator_F2S(opt.output_nc, opt.input_nc)

            if opt.cuda:
                print("Using GPU")
                self.G_A2B.cuda()
                self.G_B2A.cuda()

            self.G_A2B.apply(weights_init_normal)
            self.G_B2A.apply(weights_init_normal)
        
        Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
        self.input_A = Tensor(opt.batch_size, opt.input_nc, opt.crop_size, opt.crop_size)
        self.input_B = Tensor(opt.batch_size, opt.output_nc, opt.crop_size, opt.crop_size)
        self.target_real = torch.autograd.Variable(Tensor(opt.batch_size).fill_(1.0), requires_grad=False)
        self.target_fake = torch.autograd.Variable(Tensor(opt.batch_size).fill_(0.0), requires_grad=False)
        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

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
    
    def update_learning_rate(self):
        for lr in self.lrs:
            getattr(self, lr).step()
    

    def mask_init(self, length):
        self.mask_queue =  QueueMask(length/4)

    def set_input(self, batch):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        self.real_A = torch.autograd.Variable(self.input_A.copy_(batch['A']))
        self.real_B = torch.autograd.Variable(self.input_B.copy_(batch['B']))
        return None

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.G_A2B(self.real_A)  # G_A(A)

        self.mask_queue.insert(mask_generator(self.real_A, self.fake_B))
        
        self.rec_A = self.G_B2A(self.fake_B, self.mask_queue.last_item())   # G_B(G_A(A))
        
        self.fake_A = self.G_B2A(self.real_B, self.mask_queue.rand_item())  # G_B(B)
        
        self.rec_B = self.G_A2B(self.fake_A)   # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, self.target_real)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake,  self.target_fake)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D


    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_A = self.fake_A_buffer.push_and_pop(self.fake_A)
        self.loss_D_A = self.backward_D_basic(self.D_A, self.real_A, fake_A)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_B = self.fake_B_buffer.push_and_pop(self.fake_B)
        self.loss_D_B = self.backward_D_basic(self.D_B, self.real_B, fake_B)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.G_A2B(self.real_B)
            self.loss_idt_A = self.criterionIdentity(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.G_B2A(self.real_A)
            self.loss_idt_B = self.criterionIdentity(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A2B = self.criterionGAN(self.D_B(self.fake_B), self.target_real)

        

        # GAN loss D_B(G_B(B))
        self.loss_G_B2A = self.criterionGAN(self.D_A(self.fake_A), self.target_real)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_ABA = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_BAB = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A2B + self.loss_G_B2A + self.loss_cycle_ABA + self.loss_cycle_BAB + self.loss_idt_A + self.loss_idt_B
        
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        # self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        # self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D_A.zero_grad()   # set D_A and D_B's gradients to zero
        self.optimizer_D_B.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D_A.step()  # update D_A and D_B's weights
        self.optimizer_D_B.step()  # update D_A and D_B's weights 
