# from numpy import zeros, ones, expand_dims, asarray
# from numpy.random import randn, randint
# from keras.datasets import fashion_mnist
# from keras.optimizers import Adam
# from keras.models import Model, load_model
# from keras.layers import Input, Dense, Reshape, Flatten
# from keras.layers import Conv2D, Conv2DTranspose, Concatenate
# from keras.layers import LeakyReLU, Dropout, Embedding
# from keras.layers import BatchNormalization, Activation
# from keras import initializers
# from keras.initializers import RandomNormal
# from keras.optimizers import Adam, RMSprop, SGD
# from matplotlib import pyplot
# import numpy as np
# from math import sqrt
import sys
import tensorflow as tf
import numpy as np
import pylib as py
# print(f"Tensor Flow Version: {tf.__version__}")
# print(f"Keras Version: {tf.keras.__version__}")
# print()
# print(f"Python {sys.version}")
# gpu = len(tf.config.list_physical_devices('GPU'))>0
# print("GPU is", "available" if gpu else "NOT AVAILABLE")

output_dir = py.join('output', 'summer2winter')

b = py.glob(py.join('datasets', 'summer2winter', 'trainA'), '*.jpg')

print(output_dir)
print(b)