import time
from options.train_options import TrainOptions 
import util as tl
from models import networks, create_model
from models import create_model
import pylib as py
import numpy as np
import imlib as im
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image
from torch.autograd import Variable
from dataset.maskshadow_dataset import MaskImageDataset
import functools
import torchvision.transforms as transforms

if __name__ == '__main__':
    
    opt = TrainOptions().parse()
    model = create_model(opt)
    model.setup(opt)
    
    transforms_ = [#transforms.Resize((opt.size, opt.size), Image.BICUBIC),
        transforms.Resize(int(opt.crop_size * 1.12), Image.Resampling.BICUBIC),
        transforms.RandomCrop(opt.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    
    dataloader = DataLoader(MaskImageDataset(opt.datasets_dir, opt.dataroot, transforms_=transforms_, unaligned=True),
                batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

    plt.ioff()
    curr_iter = 0
    G_losses = []
    D_A_losses = []
    D_B_losses = []
    to_pil = transforms.ToPILImage()



    for epoch in range(opt.epoch_count, opt.epochs):

        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0
        total_iters = 0


        for i, batch in enumerate(dataloader):
            
            model.set_input(batch)
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            model.optimize_parameters()
            losses_temp = model.get_current_losses()
            print(losses_temp)
            # Set model input
            
        model.update_learning_rate()





#     G_A2B = model.G_A2B
#     G_B2A = model.G_B2A
#     D_B = model.D_B
#     D_A = model.D_A
#     g_loss_fn = model.g_loss_fn
#     d_loss_fn = model.d_loss_fn
#     cycle_loss_fn = model.cycle_loss_fn
#     identity_loss_fn = model.identity_loss_fn
#     len_dataset = model.len_dataset
#     G_lr_scheduler = networks.LinearDecay(opt.lr, opt.epochs * len_dataset, opt.epoch_decay * len_dataset)
#     D_lr_scheduler = networks.LinearDecay(opt.lr, opt.epochs * len_dataset, opt.epoch_decay * len_dataset)
#     G_optimizer = keras.optimizers.Adam(learning_rate=G_lr_scheduler, beta_1=opt.beta_1)
#     D_optimizer = keras.optimizers.Adam(learning_rate=D_lr_scheduler, beta_1=opt.beta_1)

#     ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)


    
#     @tf.function
#     def train_G(A, B):
#         with tf.GradientTape() as t:
#             A2B = G_A2B(A, training=True)
#             B2A = G_B2A(B, training=True)
#             A2B2A = G_B2A(A2B, training=True)
#             B2A2B = G_A2B(B2A, training=True)
#             A2A = G_B2A(A, training=True)
#             B2B = G_A2B(B, training=True)

#             A2B_d_logits = D_B(A2B, training=True)
#             B2A_d_logits = D_A(B2A, training=True)

#             A2B_g_loss = g_loss_fn(A2B_d_logits)
#             B2A_g_loss = g_loss_fn(B2A_d_logits)
#             A2B2A_cycle_loss = cycle_loss_fn(A, A2B2A)
#             B2A2B_cycle_loss = cycle_loss_fn(B, B2A2B)
#             A2A_id_loss = identity_loss_fn(A, A2A)
#             B2B_id_loss = identity_loss_fn(B, B2B)

#             G_loss = (A2B_g_loss + B2A_g_loss) + (A2B2A_cycle_loss + B2A2B_cycle_loss) * opt.cycle_loss_weight + (A2A_id_loss + B2B_id_loss) * opt.identity_loss_weight

#         G_grad = t.gradient(G_loss, G_A2B.trainable_variables + G_B2A.trainable_variables)
#         G_optimizer.apply_gradients(zip(G_grad, G_A2B.trainable_variables + G_B2A.trainable_variables))

#         return A2B, B2A, {'A2B_g_loss': A2B_g_loss,
#                         'B2A_g_loss': B2A_g_loss,
#                         'A2B2A_cycle_loss': A2B2A_cycle_loss,
#                         'B2A2B_cycle_loss': B2A2B_cycle_loss,
#                         'A2A_id_loss': A2A_id_loss,
#                         'B2B_id_loss': B2B_id_loss}


#     @tf.function
#     def train_D(A, B, A2B, B2A):
#         with tf.GradientTape() as t:
#             A_d_logits = D_A(A, training=True)
#             B2A_d_logits = D_A(B2A, training=True)
#             B_d_logits = D_B(B, training=True)
#             A2B_d_logits = D_B(A2B, training=True)

#             A_d_loss, B2A_d_loss = d_loss_fn(A_d_logits, B2A_d_logits)
#             B_d_loss, A2B_d_loss = d_loss_fn(B_d_logits, A2B_d_logits)
#             D_A_gp = networks.gradient_penalty(functools.partial(D_A, training=True), A, B2A, mode=opt.gradient_penalty_mode)
#             D_B_gp = networks.gradient_penalty(functools.partial(D_B, training=True), B, A2B, mode=opt.gradient_penalty_mode)

#             D_loss = (A_d_loss + B2A_d_loss) + (B_d_loss + A2B_d_loss) + (D_A_gp + D_B_gp) * opt.gradient_penalty_weight

#         D_grad = t.gradient(D_loss, D_A.trainable_variables + D_B.trainable_variables)
#         D_optimizer.apply_gradients(zip(D_grad, D_A.trainable_variables + D_B.trainable_variables))

#         return {'A_d_loss': A_d_loss + B2A_d_loss,
#                 'B_d_loss': B_d_loss + A2B_d_loss,
#                 'D_A_gp': D_A_gp,
#                 'D_B_gp': D_B_gp}


#     def train_step(A, B):
#         A2B, B2A, G_loss_dict = train_G(A, B)

#         # cannot autograph `A2B_pool`
#         A2B = model.A2B_pool(A2B)  # or A2B = A2B_pool(A2B.numpy()), but it is much slower
#         B2A = model.B2A_pool(B2A)  # because of the communication between CPU and GPU

#         D_loss_dict = train_D(A, B, A2B, B2A)

#         return G_loss_dict, D_loss_dict


#     @tf.function
#     def sample(A, B):
#         A2B = G_A2B(A, training=False)
#         B2A = G_B2A(B, training=False)
#         A2B2A = G_B2A(A2B, training=False)
#         B2A2B = G_A2B(B2A, training=False)
#         return A2B, B2A, A2B2A, B2A2B


#     # checkpoint
#     checkpoint = tl.Checkpoint(dict(G_A2B=G_A2B,
#                                     G_B2A=G_B2A,
#                                     D_A=D_A,
#                                     D_B=D_B,
#                                     G_optimizer=G_optimizer,
#                                     D_optimizer=D_optimizer,
#                                     ep_cnt=ep_cnt),
#                             py.join(model.output_dir, 'checkpoints'),
#                             max_to_keep=5)
# try:  # restore checkpoint including the epoch counter
#     checkpoint.restore().assert_existing_objects_matched()
#     print("Restored!!")
# except Exception as e:
#     print(e)
#     print("NOT restored!!")

#     # summary
#     train_summary_writer = tf.summary.create_file_writer(py.join(model.output_dir, 'summaries', 'train'))

#     # sample
#     test_iter = iter(model.A_B_dataset_test)
#     sample_dir = py.join(model.output_dir, 'samples_training')
#     py.mkdir(sample_dir)

#     # main loop
#     with train_summary_writer.as_default():
#         for ep in tqdm.trange(opt.epochs, desc='Epoch Loop'):
#             if ep < ep_cnt:
#                 continue

#             # update epoch counter
#             ep_cnt.assign_add(1)

#             # train for an epoch
#             for A, B in tqdm.tqdm(model.A_B_dataset, desc='Inner Epoch Loop', total=model.len_dataset):
#                 G_loss_dict, D_loss_dict = train_step(A, B)

#                 # # summary
#                 tl.summary(G_loss_dict, step=G_optimizer.iterations, name='G_losses')
#                 tl.summary(D_loss_dict, step=G_optimizer.iterations, name='D_losses')
#                 tl.summary({'learning rate': G_lr_scheduler.current_learning_rate}, step=G_optimizer.iterations, name='learning rate')

#                 # sample
#                 if G_optimizer.iterations.numpy() % 100 == 0:
#                     A, B = next(test_iter)
#                     A2B, B2A, A2B2A, B2A2B = sample(A, B)
#                     data_4d = np.array(A2B)
#                     data_2d = data_4d.reshape(-1, data_4d.shape[-1])
#                     np.savetxt('./yuh', data_2d, fmt='%d', delimiter='\t')

#                     img = im.immerge(np.concatenate([A, A2B, A2B2A, B, B2A, B2A2B], axis=0), n_rows=2)
#                     im.imwrite(img, py.join(sample_dir, 'iter-%09d.jpg' % G_optimizer.iterations.numpy()))

#             # save checkpoint
#             checkpoint.save(ep)