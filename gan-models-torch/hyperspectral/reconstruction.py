import numpy as np
import pysptools.distance as distance
import os
import csv
import math
from classification import Classify
from segmentation import segment_image, segmentation_patches
from pca import convert_PCA
from processor import Processor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def find_sim_patch(patch, hsi, evaluation):
    None

def reconstruction_subset(shadow_mask, feature_map):
    """
    Input:
        shadow_mask (Grayscale PIL Image): Binary mask of the segmented shadow  
        feature_map (Numpy array): 2D numpy array containing the classifications   
    Output:
        rec_from (Dict[Tuple]): Dictionary with classification and non-shadowed pixels
        rec_to (Dict[Tuple]): Dictionary with classification and shadowed pixels
    """
    shadow_mask = shadow_mask.resize((256,256), Image.BICUBIC).convert('L')
    shadow_mask = np.array(shadow_mask)
    shadow_mask = (shadow_mask > 128).astype(np.uint8)

    if shadow_mask.shape[0] != feature_map.shape[0] or shadow_mask.shape[1] != feature_map.shape[1]:

        print("Shadow mask and feature_map are not of the same dimension, resizing to 256x256")
        shadow_mask = shadow_mask.resize(256, Image.BICUBIC)
        feature_map = feature_map.resize(256, Image.BICUBIC)
        
    unique_values = np.unique(feature_map)
    rec_from = {key: [] for key in unique_values}
    rec_to = {key: [] for key in unique_values}

    for i in range(shadow_mask.shape[0]):
        for j in range(shadow_mask.shape[1]):
            if not shadow_mask[i][j]:
                rec_from[feature_map[i][j]].append((i,j))
            else:
                rec_to[feature_map[i][j]].append((i,j))
    
    return rec_from, rec_to

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

def match_patch(pixel_center, class_subset, hsi, k_means, criteria='EUD'):

    print("matching patch")

    classify = Classify(criteria)
    centerX, centerY = pixel_center
    centered_patch = hsi[centerX-1:centerX+2, centerY-1:centerY+2]
    k_means_center = k_means[centerX-1:centerX+2, centerY-1:centerY+2]


    best_score = np.iinfo(np.uint8).max
    best_candidate = None
    best_candidate_location = None
    for candidate in class_subset:
        x, y = candidate
        
        if x == 0 or x == hsi.shape[0]-1 or y == 0 or y == hsi.shape[1]-1:
            continue
        
        # print("ref", centerX, " ", centerY)
        # print('compare', x, " ", y)
        candidate_patch = hsi[x-1:x+2, y-1:y+2]
        k_means_candidate = k_means[x-1:x+2, y-1:y+2]
    
        if classify.compute_divergence_measure(k_means_center, k_means_candidate) < best_score:
            best_score = classify.compute_divergence_measure(k_means_center, k_means_candidate)
            best_candidate = candidate_patch
            best_candidate_location = (x,y)

    return best_candidate, best_candidate_location


def fine_removal(hsi, k_means, rec_from, rec_to):
    
    modified_locs = np.zeros((hsi.shape[0], hsi.shape[1]))
    refined_hsi = hsi

    for key, val in rec_to.items():
        for loc in val:
            print(loc)

            x,y = loc
            
            if x == 0 or x == hsi.shape[0]-1 or y == 0 or y == hsi.shape[1]-1:
                continue
            
            best_candidate, best_loc = match_patch(loc, rec_from[key], hsi, k_means, 'EUD')

            plot_replacement(hsi, loc, best_loc)
            refined_hsi[x-1:x+2, y-1:y+2] = best_candidate
    
    return refined_hsi

def plot_replacement(hsi, loc, candidate):
    
    patch1_center = loc  # Coordinates of the center of the first patch
    patch2_center = candidate  # Coordinates of the center of the second patch

    # Create a figure and axis
    fig, ax = plt.subplots()
    p = Processor(hsi_data=hsi)
    image = p.genFalseRGB(convertPIL=True)   
    # Display the image
    ax.imshow(image)  # Assuming grayscale image, adjust colormap if needed

    # Add rectangles around the patch locations
    rect1 = patches.Rectangle((patch1_center[0] - 1, patch1_center[1] - 1), 3, 3, linewidth=1, edgecolor='g', facecolor='none')
    rect2 = patches.Rectangle((patch2_center[0] - 1, patch2_center[1] - 1), 3, 3, linewidth=1, edgecolor='r', facecolor='none')

    # Add the rectangles to the plot
    ax.add_patch(rect1)
    ax.add_patch(rect2)

    # Set axis labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Image with Patches')

    # Show the plot
    plt.show()


def refine_boundaries(hsi, shadow_truth, threshold = 100):
    None




p = Processor()
hsi = p.prepare_data(r'datasets/export_2/trainA/session_000_001k_048_snapshot_ref.tiff')
#p.genFalseRGB(visualize=True)
endmember_data = None
with open(r'datasets/export_2/endmembers/endmembers.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    endmember_data = {col: [] for col in reader.fieldnames}

    for row in reader:
        for col in reader.fieldnames:
            try:
                value = int(row[col])
            except ValueError:
                # Handle the case where the value cannot be converted to an integer
                # You might want to handle this differently based on your requirements
                value = 0  # Default value if conversion fails

            endmember_data[col].append(value)

feature_map = segment_image(hsi, endmember_data)
segmentation_patches(feature_map)

shadow_mask = Image.open(r'datasets/shadow_masks/48_mask.png')
shadow_mask = shadow_mask.resize((256,256), Image.BICUBIC)#.convert('L')
orig = p.genFalseRGB(convertPIL=True)

shadow_mask = shadow_mask.resize((256, 256))
orig = orig.resize((256, 256))

print(orig.size)
print(shadow_mask.size)

# blended = Image.blend(orig, shadow_mask, 0.5)

# # Display the blended image
# blended.show()

rec_from, rec_to = reconstruction_subset(shadow_mask, feature_map)

pca_scores = convert_PCA(hsi, 3, False)
print(pca_scores)

refined = fine_removal(hsi, pca_scores, rec_from, rec_to)

p2 = Processor(hsi_data=refined)

orig_image = p.genFalseRGB(convertPIL=True)
rec_image = p2.genFalseRGB(convertPIL=True)

rec_image.save(r'datasets/shadow_masks/reconstructed.png')

orig_image.show()
rec_image.show()
shadow_mask.show()

# Alternatively, you can display them using matplotlib
fig, axes = plt.subplots(1, 3)  # Create a figure with 1 row and 3 columns
axes[0].imshow(orig_image)
axes[0].axis('off')  # Turn off axis
axes[1].imshow(rec_image)
axes[1].axis('off')
axes[2].imshow(shadow_mask)
axes[2].axis('off')
plt.show()