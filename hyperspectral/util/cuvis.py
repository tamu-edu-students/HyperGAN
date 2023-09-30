import os # File system operation
import numpy as np # Load numeric data operations
import cuvis # Load the Cubert SDK
import typing # Typehints for python data
import traceback
import matplotlib.pyplot as plt # Visualization of images
#from cuvis import Measurement

# lib_dir = os.getenv("CUVIS")
# default_path = os.path.normpath(os.path.join(lib_dir, os.path.pardir, "sdk", "sample_data", "set1"))
# fc = FileChooser(r"C:\Users\vamin\OneDrive\Desktop\CAST\code\HyperGAN\Dataset\lentil.tiff")
# print(type(fc))
c = cuvis()

### Create a Measurement from the Image
try:
    mesu = c.Measurement(r"C:\Users\vamin\OneDrive\Desktop\CAST\code\HyperGAN\Dataset\lentil.tiff")
except Exception as e:
    print(traceback.print_exc())