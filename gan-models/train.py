import tensorflow as tf
import time
from options.train_options import TrainOptions 
import util as tl
from models import networks, create_model
from models import create_model
import tqdm
import pylib as py
import numpy as np
import imlib as im

if __name__ == '__main__':
    opt = TrainOptions().parse()
    model = create_model(opt)
    
    ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

# checkpoint
    checkpoint = tl.Checkpoint(dict(G_A2B=model.G_A2B,
                                G_B2A=model.G_B2A,
                                D_A=model.D_A,
                                D_B=model.D_B,
                                G_optimizer=model.G_optimizer,
                                D_optimizer=model.D_optimizer,
                                ep_cnt=ep_cnt),
                           py.join(model.output_dir, 'checkpoints'),
                           max_to_keep=5)
    try:  # restore checkpoint including the epoch counter
        checkpoint.restore().assert_existing_objects_matched()
    except Exception as e:
        print(e)

    # summary
    train_summary_writer = tf.summary.create_file_writer(py.join(model.output_dir, 'summaries', 'train'))

    # sample
    test_iter = iter(model.A_B_dataset_test)
    sample_dir = py.join(model.output_dir, 'samples_training')
    py.mkdir(sample_dir)

    # main loop
    with train_summary_writer.as_default():
        for ep in tqdm.trange(opt.epochs, desc='Epoch Loop'):
            if ep < ep_cnt:
                continue

            # update epoch counter
            ep_cnt.assign_add(1)

            # train for an epoch
            for A, B in tqdm.tqdm(model.A_B_dataset, desc='Inner Epoch Loop', total=model.len_dataset):
                G_loss_dict, D_loss_dict = model.optimize_parameters(A, B, opt)

                # # summary
                tl.summary(G_loss_dict, step=model.G_optimizer.iterations, name='G_losses')
                tl.summary(D_loss_dict, step=model.G_optimizer.iterations, name='D_losses')
                tl.summary({'learning rate': model.G_lr_scheduler.current_learning_rate}, step=model.G_optimizer.iterations, name='learning rate')

                # sample
                if model.G_optimizer.iterations.numpy() % 100 == 0:
                    A, B = next(test_iter)
                    A2B, B2A, A2B2A, B2A2B = model.sample(A, B)
                    img = im.immerge(np.concatenate([A, A2B, A2B2A, B, B2A, B2A2B], axis=0), n_rows=2)
                    im.imwrite(img, py.join(sample_dir, 'iter-%09d.jpg' % model.G_optimizer.iterations.numpy()))

            # save checkpoint
            checkpoint.save(ep)