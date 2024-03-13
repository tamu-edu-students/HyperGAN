import argparse
import torch
from .extractor import ViTExtractor
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import cv2
import math
from PIL import Image
from .classification import Classify, Objective
from .processor import Processor
from scipy.ndimage import zoom
import tifffile
from .util.eval_metrics import calculate_rmse, SSIM
from .util.utils import hyper_measures_eval

def chunk_cosine_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ Computes cosine similarity between all possible pairs in two sets of vectors.
    Operates on chunks so no large amount of GPU RAM is required.
    :param x: an tensor of descriptors of shape Bx1x(t_x)xd' where d' is the dimensionality of the descriptors and t_x
    is the number of tokens in x.
    :param y: a tensor of descriptors of shape Bx1x(t_y)xd' where d' is the dimensionality of the descriptors and t_y
    is the number of tokens in y.
    :return: cosine similarity between all descriptors in x and all descriptors in y. Has shape of Bx1x(t_x)x(t_y) """
    result_list = []
    num_token_x = x.shape[2]
    for token_idx in range(num_token_x):
        token = x[:, :, token_idx, :].unsqueeze(dim=2)  # Bx1x1xd'
        result_list.append(torch.nn.CosineSimilarity(dim=3)(token, y))  # Bx1xt
    return torch.stack(result_list, dim=2)  # Bx1x(t_x)x(t_y)


def shadow_sim_iter(rgb_image, hsi_data, reconstruction_set, shadow_map, load_size: int = 224, layer: int = 11,
                                facet: str = 'key', bin: bool = False, stride: int = 4, model_type: str = 'dino_vits8'):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = ViTExtractor(model_type, stride, device=device)
    extractor.model = extractor.patch_vit_resolution(extractor.model, 4)
    patch_size = extractor.model.patch_embed.patch_size

    image_batch_a, image_pil_a = extractor.preprocess(rgb_image, load_size)
    print(image_batch_a.shape)
    image_batch_b, image_pil_b = extractor.preprocess(rgb_image, load_size)
    descs_a = extractor.extract_descriptors(image_batch_a.to(device), layer, facet, bin, include_cls=True)
    descs_b = extractor.extract_descriptors(image_batch_b.to(device), layer, facet, bin, include_cls=True)
    similarities = chunk_cosine_sim(descs_a, descs_b)

    hsi_refined = hsi_data

    for i, j in reconstruction_set:
        best_candidate_loc, center_orig = similarity_finder(i,j,similarities, shadow_map, hsi_data, extractor, num_sim_patches=30)
        hsi_refined = replace_spectra(hsi_refined, center_orig, best_candidate_loc)
        
    return hsi_refined


def similarity_finder(x_coor, y_coor, similarities, shadow_map, hsi_data, extractor, stride: int = 4, num_sim_patches: int = 1):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(x_coor, ": xcoordinate ",y_coor , " : ycoordinate" )
    patch_size = extractor.model.patch_embed.patch_size
    num_patches_a, load_size_a = extractor.num_patches, extractor.load_size
    num_patches_b, _ = extractor.num_patches, extractor.load_size
    #print("num patches", num_patches_b)

    new_H = patch_size / stride * (load_size_a[0] // patch_size - 1) + 1
    new_W = patch_size / stride * (load_size_a[1] // patch_size - 1) + 1
    y_descs_coor = int(new_H / load_size_a[0] * y_coor) + 1
    x_descs_coor = int(new_W / load_size_a[1] * x_coor) + 1

    orig_x_patch_coord = x_descs_coor
    orig_y_patch_coord = y_descs_coor
    #print("patch coord", (orig_x_patch_coord, orig_y_patch_coord))

    center_orig = ((x_descs_coor - 1) * stride + stride + patch_size // 2 - .5,
                (y_descs_coor - 1) * stride + stride + patch_size // 2 - .5)
    
    raveled_desc_idx = num_patches_a[1] * y_descs_coor + x_descs_coor
    reveled_desc_idx_including_cls = raveled_desc_idx + 1
    curr_similarities = similarities[0, 0, reveled_desc_idx_including_cls, 1:]
    curr_similarities = curr_similarities.reshape(num_patches_b)

    print(curr_similarities.shape)
    print("and", shadow_map.flatten().shape)

    shadow_map = torch.tensor(shadow_map).to(device)
    #Subtract shadow map binary values, after converting to torch tensor so that similat
    sims, idxs = torch.topk(curr_similarities.flatten()-10*shadow_map.flatten(), num_sim_patches)
    print("similarities length", sims.shape, "and they are", sims)
    # print("index length", idxs.shape, "and they are", idxs)
    # mask_filtered_indices = idxs[shadow_map.flatten() == 0]
    # top_5_matches = mask_filtered_indices[:num_sim_patches]
    box_coords = []
    for idx, sim in zip(idxs, sims):
        y_descs_coor, x_descs_coor = idx // num_patches_b[1], idx % num_patches_b[1]
        center = ((x_descs_coor - 1) * stride + stride + patch_size // 2 - .5,
                    (y_descs_coor - 1) * stride + stride + patch_size // 2 - .5)
        box_coords.append(center)
    
    #return box_coords
        
    floor_center_orig = tuple(math.floor(val) for val in center_orig)
    box_coords = np.array([[math.floor(val) for val in tpl] for tpl in box_coords])

    print("center orig is", center_orig)
    _, best_candidate_loc = spectral_replacement_evaluator(hsi_data, floor_center_orig, box_coords)
    print("best candidate found to be ", best_candidate_loc)
    # fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    # shadow_mas = shadow_map.cpu().numpy()
    # # Plot the first image (leftmost)
    # axes[0].imshow(shadow_mas, cmap='gray', interpolation='nearest')
    # axes[0].set_title('Image 1')
    # x_orig, y_orig = center_orig
    # #center_orig = (center_orig[0].cpu(),center_orig[1].cpu())
    # print(center_orig, "to check")
    # axes[0].add_patch(plt.Rectangle((orig_x_patch_coord-0.5, orig_y_patch_coord-0.5), 1, 1, linewidth=2, edgecolor='b', facecolor='none'))

    # # Plot the second image (middle)
    # axes[1].imshow((curr_similarities.flatten()-shadow_map.flatten()).reshape(num_patches_b).cpu().numpy(), cmap='jet')
    # axes[1].set_title('Image 2')

    # # Plot the third image (rightmost)
    # origimage = Image.open('datasets/export_3/trainA/13_rgb.png')
    # # Resize the image
    # resized_image = origimage.resize((236, 224), Image.BICUBIC)
    # axes[2].imshow(resized_image)
    # axes[2].set_title('Image 3')
    
    # best_x, best_y = best_candidate_loc
    # # Draw a box on the first image at specified coordinates
    # for x, y in box_coords:
    #     # x = x.cpu()
    #     # y = y.cpu()
    #     axes[2].add_patch(plt.Rectangle((x - 2, y - 2), 4, 4, linewidth=2, edgecolor='r', facecolor='none'))
    #     axes[2].add_patch(plt.Rectangle((x_orig-2, y_orig-2), 4, 4, linewidth=2, edgecolor='b', facecolor='none'))
    #     axes[2].add_patch(plt.Rectangle((best_x-2, best_y-2), 4, 4, linewidth=2, edgecolor='g', facecolor='none'))

    # shadow_im = Image.open('datasets/shadow_masks/13_mask.png')
    # shadow_im = shadow_im.resize((236,224), Image.BICUBIC)
    # axes[3].imshow(shadow_im)
    # axes[3].set_title('Image 4')
    # axes[3].add_patch(plt.Rectangle((x_orig-2, y_orig-2), 4, 4, linewidth=2, edgecolor='b', facecolor='none'))

    # # Hide axis for all subplots
    # for ax in axes:
    #     ax.axis('off')

    # plt.tight_layout()
    # plt.show()

    return best_candidate_loc, floor_center_orig


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

def spectral_replacement_evaluator(hsi, orig_loc, patches_loc, stride=4):
    
    classifier = Classify(evaluation='SCM')
    
    center_x, center_y = orig_loc
    start_x = center_x - stride // 2
    start_y = center_y - stride // 2
    
    orig_patch = hsi[start_y:start_y+stride, start_x:start_x+stride,  :]
    candidate_patches = []
    for x,y in patches_loc:
        
        candidate_test_x = x - stride // 2
        candidate_test_y = y - stride // 2

        print("patch location:", (candidate_test_y, candidate_test_y+stride), "amd", (candidate_test_x, candidate_test_x+stride))

        candidate_patches.append(hsi[candidate_test_y:candidate_test_y+stride, candidate_test_x:candidate_test_x+stride, :])
            
    return classifier.find_best_patch(orig_patch, candidate_patches, patches_loc)

def replace_spectra(hsi, orig_loc, candidate_loc, patch_size=4):
    
    center_x, center_y = orig_loc
    start_x = center_x - patch_size // 2
    start_y = center_y - patch_size // 2

    candidate_center_x, candidate_center_y = candidate_loc
    candidate_start_x = candidate_center_x - patch_size // 2
    candidate_start_y = candidate_center_y - patch_size // 2

    print("SIZEEEE", hsi.shape)

    hsi[start_y:start_y+patch_size, start_x:start_x+patch_size, :] = hsi[candidate_start_y:candidate_start_y+patch_size, candidate_start_x:candidate_start_x+patch_size, :]

    return hsi

def blur_where_mask(image, mask):
    """
    Blur the regions in the image where the mask value is 1.
    
    Parameters:
        image (numpy.ndarray): Input image.
        mask (numpy.ndarray): Binary mask indicating regions to blur (1 for regions to blur, 0 otherwise).
    
    Returns:
        numpy.ndarray: Blurred image.
    """
    # Create a copy of the input image to store the result
    blurred_image = image.copy()
    
    # Iterate over each pixel in the image
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            # Check if the mask value at this location is 1
            if mask[y, x] == 1:
                # Apply Gaussian blur to the 3x3 neighborhood around the pixel
                roi = image[max(0, y-1):min(y+2, image.shape[0]), max(0, x-1):min(x+2, image.shape[1])]
                blurred_roi = cv2.GaussianBlur(roi, (3, 3), 0)
                # Update the corresponding region in the blurred image
                blurred_image[max(0, y-1):min(y+2, image.shape[0]), max(0, x-1):min(x+2, image.shape[1])] = blurred_roi
    
    return blurred_image


def fine_removal(shadow_mask, orig_image, gan_output):

    """
    shadow_mask (PIL Image) - shadow mask of the image
    orig_image (numpy array) - unresolved hsi image before gan
    gan_output (numpy array) - coarsely refined hsi image
    
    """

    #shadow patch processing
    shadow_mask = shadow_mask.convert('L')
    shadow_mask = np.array(shadow_mask.resize((236,224), Image.BICUBIC))
    patch_mask, rec_indxs = shadow_patches(shadow_mask)

    # Upsample the array using bicubic interpolation, for display and matching
    upsampled_array = np.uint8(zoom(patch_mask, (224 / patch_mask.shape[0], 236 / patch_mask.shape[1]), order=3))

    #formatting orig data
    origP = Processor(hsi_data=orig_image)
    shadow_im = origP.hyperCrop2D(origP.hsi_data, 224, 236)

    #formatting gan data
    ganP = Processor(hsi_data=gan_output)
    hsi_data = ganP.hyperCrop2D(ganP.hsi_data, 224, 236)
    init_image = ganP.genFalseRGB(convertPIL=True)
    init_image = init_image.resize((290, 275))

    #fine removal
    start_time = time.time()
    hsi_resolved = None
    with torch.no_grad():
        hsi_resolved = shadow_sim_iter(init_image, hsi_data, rec_indxs, patch_mask)

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time} seconds")

    fineP = Processor(hsi_resolved)
    final_image = fineP.genFalseRGB(convertPIL=True)

    #plot results
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    axes[0].imshow(origP.genFalseRGB(convertPIL=True))
    axes[1].imshow(init_image)
    axes[2].imshow(fineP.genFalseRGB(convertPIL=True))
    axes[3].imshow(upsampled_array)

    plt.tight_layout()
    #plt.show()

    return hsi_resolved


if __name__ == '__main__':
    
    shadow_mask = Image.open(r'datasets/ctf_pipeline/mask/20.png').convert('L')
    shadow_mask = shadow_mask.resize((236,224), Image.BICUBIC)
    # plt.imshow(shadow_mask, cmap='gray')
    # plt.show()
    shadow_mask = np.array(shadow_mask)
    patch_mask, rec_indxs = shadow_patches(shadow_mask)

    with tifffile.TiffFile(r'datasets/ctf_pipeline/testA/20.tiff') as tif:
        shadow_image = tif.asarray()

    shadowP = Processor()
    shadowP.prepare_data(r'datasets/ctf_pipeline/testA/20.tiff')
    shadow_im = shadowP.hyperCrop2D(shadowP.hsi_data, 224, 236)

    with tifffile.TiffFile('datasets/ctf_pipeline/coarse/20.tiff') as tif:
        image = tif.asarray()  # Convert the TIFF image to a numpy array
    p = Processor(hsi_data=image)
    #hsi_data = p.prepare_data(r'datasets/export_2/trainA/session_000_001k_048_snapshot_ref.tiff')
    hsi_data = p.hyperCrop2D(p.hsi_data, 224, 236)
    init_image = p.genFalseRGB(convertPIL=True)

    # plt.imshow(patch_mask, cmap='gray', interpolation='nearest')
    # plt.axis('off')  # Turn off axis
    # plt.show()
    rgb_image_path = r'datasets/shadow_masks/48_gan.png'
    img = init_image #Image.open(rgb_image_path)
    resized_img = img.resize((290, 275))  # Specify the new dimensions\
    resized_img.save(rgb_image_path)

    start_time = time.time()
    hsi_resolved = None
    with torch.no_grad():
        hsi_resolved = shadow_sim_iter(resized_img, hsi_data, rec_indxs, patch_mask)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time} seconds")

    p2 = Processor(hsi_resolved)
    final_image = p2.genFalseRGB(convertPIL=True)

    tifffile.imsave(r'datasets/shadow_masks/resolved.tiff', hsi_resolved)

    final_image = np.array(final_image)
    
    # Apply Sobel filter in x and y directions
    sobelx = cv2.Sobel(final_image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(final_image, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    # Normalize the gradient to convert to 8-bit image
    gradient_normalized = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


    sobel_channels = []
    for channel in range(hsi_resolved.shape[2]):
        sobelx = cv2.Sobel(hsi_resolved[:,:, channel], cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(hsi_resolved[:,:, channel], cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        sobel_channels.append(gradient_magnitude)

    # Merge the channels back into a single multispectral image
    sobel_multispectral_image = cv2.merge(sobel_channels)

    sobel_gradient_combined = np.mean(sobel_channels, axis=0)

    # # Display the gradient image
    # cv2.imshow("Gradient Image", gradient_normalized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print(patch_mask.shape[0])
    upsampling_factor_row = 224 / patch_mask.shape[0]
    upsampling_factor_col = 236 / patch_mask.shape[1]

    # Upsample the array using bicubic interpolation
    upsampled_array = zoom(patch_mask, (upsampling_factor_row, upsampling_factor_col), order=3)

    # Convert the upsampled array to uint8 if needed
    upsampled_array = np.uint8(upsampled_array)

    mask = ((sobel_gradient_combined*upsampled_array)>50).astype(np.uint8)

    p3 = Processor()
    p3.prepare_data(r'datasets/ctf_pipeline/ref/20.tiff')
    hsi_data3 = p3.hyperCrop2D(p3.hsi_data, 224, 236)

    



    with tifffile.TiffFile('datasets/ctf_pipeline/coarse/20.tiff') as tif:
        image = tif.asarray()  # Convert the TIFF image to a numpy array
    p = Processor(hsi_data=image)
    #hsi_data = p.prepare_data(r'datasets/export_2/trainA/session_000_001k_048_snapshot_ref.tiff')
    hsi_data_5 = p.hyperCrop2D(p.hsi_data, 224, 236)

    o = Objective()

    print("Init improvement")
    print(o.SID(hsi_data_5, hsi_data3))
    print("Final improvement")
    print(o.SID(hsi_resolved, hsi_data3))

    print("Trying this")
    print(calculate_rmse(hsi_data_5, hsi_data3))
    print("and")
    print(calculate_rmse(hsi_resolved, hsi_data3))

    
    rmse, ssim = hyper_measures_eval(hsi_data3, shadow_im, rec_indxs)
    print("mask specific results", rmse, " and ", ssim)

    rmse, ssim = hyper_measures_eval(hsi_data3, hsi_data_5, rec_indxs)
    print("mask coarse specific results", rmse, " and ", ssim)

    rmse, ssim = hyper_measures_eval(hsi_data3, hsi_resolved, rec_indxs)
    print("mask fine specific results", rmse, " and ", ssim)
    # print("sanity check")
    # print(SSIM(hsi_data, hsi_resolved))
    # print(mask.shape)
    # print(type(mask[6,6]))

    # print(final_image.shape)
    # print(type(final_image[6,6,0]))

    
    # blurred_image = cv2.GaussianBlur(final_image, (5, 5), 0)
    # smoothed_image = cv2.bitwise_and(final_image, final_image, mask=(1 - mask))

    # # Combine the original and smoothed images
    # result_image = cv2.add(blurred_image, smoothed_image)

    blur_image = blur_where_mask(final_image, mask)

    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    axes[0].imshow(shadowP.genFalseRGB(convertPIL=True))
    axes[1].imshow(init_image)
    axes[2].imshow(gradient_normalized)
    axes[3].imshow(blur_image)
    axes[4].imshow(upsampled_array)
    


    plt.tight_layout()
    plt.show()

    # # Define a threshold for high gradients
    # gradient_threshold = 100  # Adjust this threshold as needed

    # # Create a mask where high gradients are present
    # high_gradient_mask = (gradient_magnitude > gradient_threshold).astype(np.uint8)

    # # Apply Gaussian blur only to areas with high gradients
    # blurred_image = cv2.GaussianBlur(final_image, (5, 5), 0)
    # smoothed_image = cv2.bitwise_and(final_image, final_image, mask=(1 - high_gradient_mask))

    # # Combine the original and smoothed images
    # result_image = cv2.add(blurred_image, smoothed_image)

    # # Display the result
    # cv2.imshow("Result", result_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()