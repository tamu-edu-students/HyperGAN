# HyperGAN
Tools and development for creating an unsupervised GAN for hyperspectral imaging shadow compensation to increase performance for Autonomous Vehicles. Work conducted as a part of the Connected Autonomous Safe Technologies (CAST) Group.


cd into the Docker folder and issue this command to build a docker image:

`docker build -t hypergan_cuda -f Dockerfile_cubert_cuda .`

Have HyperGAN and a directory called HyperImages at the same level, HyperImages is not currently used for training but will be soon. 

Once the image is built, reopen the code in a container

Download the datasets folder from https://drive.google.com/drive/folders/1uIKsODrQ0znt1ITwYGtSFLMZcOm1ItUP?usp=drive_link
Download the outputs folder from https://drive.google.com/drive/folders/18rzZWJ-cCBqwpNb3z-Qs3Bhu-3hJqxdD?usp=drive_link

Move both of these directories into HyperGAN. 

HyperGAN/datasets/ is composed of training datasets (such as shadow_USR) with subdirectories of TrainA, TrainB, TestA, and TestB. The code will know which folders to access based on this organization. HyperGAN/output consists of results from each dataset. Checkpoints are models that are labeled epoch_(epoch number) where epoch number is how many epochs the model has been trained for. Model progress is saved at the end of each epoch. The models are sampled every 100 iterations during training, and their output is shown in HyperGAN/output/shadow_USR/samples_training. For testing, each image in testA will be deshadowed, and they will be output in HyperGAN/output/shadow_USR/samples_testing.


To run training, cd into HyperGAN and enter this command

`python gan-models-torch/train.py --dataroot shadow_USR --model maskshadow_gan`

if your training stopped randomly, enter this command (let's say it stopped at epoch 94)

`python gan-models-torch/train.py --dataroot shadow_USR --model maskshadow_gan --restore --epoch_count 94`

To test a model at a certain epoch checkpoint (say epoch 134), cd into HyperGAN and enter this command.

`python gan-models-torch/test.py --dataroot shadow_USR --model maskshadow_gan --epoch_count 134`


