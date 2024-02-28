import argparse
import torch
from extractor import ViTExtractor
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import cv2
from PIL import Image

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


def shadow_sim_iter(image_path: str, reconstruction_set, shadow_map, load_size: int = 224, layer: int = 11,
                                facet: str = 'key', bin: bool = False, stride: int = 4, model_type: str = 'dino_vits8'):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = ViTExtractor(model_type, stride, device=device)
    extractor.model = extractor.patch_vit_resolution(extractor.model, 4)

    image_batch_a, image_pil_a = extractor.preprocess(image_path, load_size)
    print(image_batch_a.shape)
    image_batch_b, image_pil_b = extractor.preprocess(image_path, load_size)
    descs_a = extractor.extract_descriptors(image_batch_a.to(device), layer, facet, bin, include_cls=True)
    descs_b = extractor.extract_descriptors(image_batch_b.to(device), layer, facet, bin, include_cls=True)
    similarities = chunk_cosine_sim(descs_a, descs_b)


    for i, j in reconstruction_set:
        similarities_map(i,j,similarities, shadow_map, extractor, num_sim_patches=7)



def similarities_map(x_coor, y_coor, similarities, shadow_map, extractor, stride: int = 4, num_sim_patches: int = 1):
    
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

    shadow_map = torch.tensor(shadow_map).to(device)
    #Subtract shadow map binary values, after converting to torch tensor so that similat
    sims, idxs = torch.topk(curr_similarities.flatten()-10*shadow_map.flatten(), num_sim_patches)
    #print("similarities length", sims.shape, "and they are", sims)
    # print("index length", idxs.shape, "and they are", idxs)
    # mask_filtered_indices = idxs[shadow_map.flatten() == 0]
    # top_5_matches = mask_filtered_indices[:num_sim_patches]
    box_coords = []
    for idx, sim in zip(idxs, sims):
        y_descs_coor, x_descs_coor = idx // num_patches_b[1], idx % num_patches_b[1]
        center = ((x_descs_coor - 1) * stride + stride + patch_size // 2 - .5,
                    (y_descs_coor - 1) * stride + stride + patch_size // 2 - .5)
        box_coords.append(center)

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
    
    # print(type(box_coords[0][0]))
    # # Draw a box on the first image at specified coordinates
    # for x, y in box_coords:
    #     x = x.cpu()
    #     y = y.cpu()
    #     axes[2].add_patch(plt.Rectangle((x - 2, y - 2), 4, 4, linewidth=2, edgecolor='r', facecolor='none'))
    #     axes[2].add_patch(plt.Rectangle((x_orig-2, y_orig-2), 4, 4, linewidth=2, edgecolor='b', facecolor='none'))

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


if __name__ == '__main__':
    
    shadow_mask = Image.open(r'datasets/shadow_masks/13_mask.png').convert('L')
    shadow_mask = shadow_mask.resize((236,224), Image.BICUBIC)
    # plt.imshow(shadow_mask, cmap='gray')
    # plt.show()
    patch_mask, rec_indxs = shadow_patches(np.array(shadow_mask))

    plt.imshow(patch_mask, cmap='gray', interpolation='nearest')
    plt.axis('off')  # Turn off axis
    plt.show()


    # # Define the coordinates of the box
    # x_start, x_end = 205, 215
    # y_start, y_end = 140, 160

    # pixel_array = np.empty((y_end - y_start + 1, x_end - x_start + 1), dtype=object)

    # for y in range(y_start, y_end + 1):
    #     for x in range(x_start, x_end + 1):
    #         pixel_array[y - y_start, x - x_start] = (x, y)
    
    start_time = time.time()


    with torch.no_grad():
        shadow_sim_iter(r'datasets/export_3/trainA/13_rgb.png', rec_indxs, patch_mask)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time} seconds")
