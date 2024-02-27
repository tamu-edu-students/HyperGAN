import numpy as np
import pysptools.distance as distance
import os
import csv
import math
from classification import Classify
from processor import Processor
import matplotlib.pyplot as plt

def segment_image(hsi, endmembers, similarity_measure='SCM', visualize=True):

    """
    inputs:
        hsi (np.array) - hyperspectral cube as a 3D numpy array
        endmembers (dict) - dictionary where the label is the endmember name, and the value is the spectral curve
        simlarity measure (string) - 
    
    """
    classify = Classify(similarity_measure)
    im_classification = np.empty((hsi.shape[0], hsi.shape[1]), dtype=object)

    for i in range(hsi.shape[0]):
        for j in range(hsi.shape[1]):
            im_classification[i][j] = classify.classify_by_min(endmembers, hsi[i][j])

    print(im_classification)

    unique_values, counts = np.unique(im_classification, return_counts=True)

    # Print the unique values and their counts
    for value, count in zip(unique_values, counts):
        print(f"Value: {value}, Count: {count}")


    #array_2d = np.random.choice(['foreground', 'white_background', 'tree'], size=(275, 290))

    value_map = {'foreground': 0, 'white_background': 1, 'tree': 2}

    # Creating a numerical array
    numerical_array = np.array([[value_map[value] for value in row] for row in im_classification])

    if visualize:
        # Plotting the numerical array with a colormap
        plt.imshow(numerical_array, cmap='viridis', interpolation='nearest')
        plt.colorbar(ticks=[0, 1, 2], label='Value')  # Add a colorbar with labels
        plt.axis('off')  # Turn off the axis
        plt.show()

    return numerical_array

