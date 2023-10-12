# HyperGAN
tools and development for creating a GAN to improve hyperspectral imaging performance


cd into the Docker folder and issue this command to build a docker image:

`docker build -t hypergan_cuda -f Dockerfile_cubert_cuda .`

Have HyperGAN and a directory called HyperImages at the same level, HyperImages is not currently used for training but will be soon. 

Once the image is built, reopen the code in a container

Download the datasets folder from https://drive.google.com/drive/folders/1uIKsODrQ0znt1ITwYGtSFLMZcOm1ItUP?usp=drive_link
Download the outputs folder from https://drive.google.com/drive/folders/18rzZWJ-cCBqwpNb3z-Qs3Bhu-3hJqxdD?usp=drive_link

Move both of these directories into HyperGAN. 

Datasets/ is composed of training datasets with subdirectories of TrainA, TrainB, TestA, and TestB. The code will know which folders to access based on this organization. The output folder consists of results from each dataset. Checkpoints are models that are labeled epoch_(epoch number) where epoch number is how many epochs the model has been trained for. Model progress is saved at the end of each epoch. The models are sampled every 100 iterations during training, and its output is shown in samples_training. For testing, each image in testA will be deshadowed, and it will be output in samples_testing.


To run training, cd into HyperGAN and enter this command

`python gan-models-torch/train.py --dataroot shadow_USR --model maskshadow_gan`

if your training stopped randomly, enter this command (lets say it stopped at epoch 94)

`python gan-models-torch/train.py --dataroot shadow_USR --model maskshadow_gan --restore --epoch_count 94`

To test a model at a certain epoch checkpoint (say epoch 134), cd into HyperGAN and enter this command.

`python gan-models-torch/test.py --dataroot shadow_USR --model maskshadow_gan --epoch_count 134`


