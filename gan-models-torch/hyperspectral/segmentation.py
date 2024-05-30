import numpy as np
import pysptools.distance as distance
import os
import csv
import math
from classification import Classify
from processor import Processor
import matplotlib.pyplot as plt

def segment_image(hsi, endmembers, similarity_measure='SCM', visualize=False):

    """
    inputs:
        hsi (np.array) - hyperspectral cube as a 3D numpy array
        endmembers (dict) - dictionary where the label is the endmember name, and the value is the spectral curve
        simlarity measure (string) - spectral similarity algorithm used to segment image
    
    """
    classify = Classify(similarity_measure)
    im_classification = np.empty((hsi.shape[0], hsi.shape[1]), dtype=np.uint8)

    for i in range(hsi.shape[0]):
        for j in range(hsi.shape[1]):
            im_classification[i][j] = classify.classify_by_min(endmembers, hsi[i][j])

    unique_values, counts = np.unique(im_classification, return_counts=True)

    # Print the unique values and their counts
    for value, count in zip(unique_values, counts):
        print(f"Value: {value}, Count: {count}")

    if visualize:
        # Plotting the numerical array with a colormap
        plt.imshow(im_classification, cmap='viridis', interpolation='nearest')
        plt.colorbar(ticks=[0, 1, 2], label='Value')  # Add a colorbar with labels
        plt.axis('off')  # Turn off the axis
        plt.show()

    return im_classification

def segment_patches(segmentation_map, patch_size=4):
    height, width = segmentation_map.shape
    
    segmented_patches = np.zeros(( (height//patch_size)-1, (width//patch_size)-1 ), dtype=np.uint8)

        # Iterate over the image in patch_size steps
    for y in range(0, height-patch_size, patch_size):
        for x in range(0, width-patch_size, patch_size):
            
            patch = segmentation_map[y:y+patch_size, x:x+patch_size]
            replacement_val = round(np.mean(patch))
            segmented_patches[y // patch_size, x // patch_size] = replacement_val

    plt.imshow(segmentation_map, cmap='viridis', interpolation='nearest')
    plt.colorbar(ticks=[0, 1, 2], label='Value')  # Add a colorbar with labels
    plt.axis('off')  # Turn off the axis
    #plt.show()

    return segmented_patches

def shadow_patches(shadow_mask, patch_size=4, threshold=128):

    height, width = shadow_mask.shape
    shadow_indxs = []
    # Initialize an empty binary image
    binary_image = np.zeros(( (height//patch_size)-1, (width//patch_size)-1 ), dtype=np.uint8)

    # Iterate over the image in patch_size steps
    for y in range(0, height-patch_size, patch_size):
        for x in range(0, width-patch_size, patch_size):
            
            patch = shadow_mask[y:y+patch_size, x:x+patch_size]
            patch_mean = np.mean(patch)

            if patch_mean > threshold/32:
                binary_image[y // patch_size, x // patch_size] = 1
                shadow_indxs.append((x, y))
                print("shadow index", (x // patch_size, y // patch_size))
    
    print(binary_image.shape)
    return binary_image, shadow_indxs

def lightness_mapping(hsi, method='geo_avg'):
    """
    inputs:
        hsi (np.array) - hyperspectral cube as a 3D numpy array
        method (str) - method for producing lightness map
                            geo_avg - geometric mean
                            eucl_norm - euclidean norm
                            avg - average
    """
    None