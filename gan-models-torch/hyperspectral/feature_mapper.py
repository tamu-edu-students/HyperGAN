import argparse
import torch
from extractor import ViTExtractor
import numpy as np
import matplotlib.pyplot as plt
from inspect_similarity import chunk_cosine_sim
import time

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


def shadow_sim_iter(image_path: str, reconstruction_set, load_size: int = 224, layer: int = 11,
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

    iter = 0
    for i, j in reconstruction_set:
        #iter+=1
        #print(iter)
        similarities_map(i,j,similarities, extractor, num_sim_patches=7)

def similarities_map(x_coor, y_coor, similarities, extractor, stride: int = 4, num_sim_patches: int = 1):
    
    patch_size = extractor.model.patch_embed.patch_size
    num_patches_a, load_size_a = extractor.num_patches, extractor.load_size
    num_patches_b, load_size_b = extractor.num_patches, extractor.load_size

    new_H = patch_size / stride * (load_size_a[0] // patch_size - 1) + 1
    new_W = patch_size / stride * (load_size_a[1] // patch_size - 1) + 1
    y_descs_coor = int(new_H / load_size_a[0] * y_coor)
    x_descs_coor = int(new_W / load_size_a[1] * x_coor)



    # draw chosen point
    center = ((x_descs_coor - 1) * stride + stride + patch_size // 2 - .5,
                (y_descs_coor - 1) * stride + stride + patch_size // 2 - .5)
    
    #print('chosen point', center)
    # get and draw current similarities
    raveled_desc_idx = num_patches_a[1] * y_descs_coor + x_descs_coor
    reveled_desc_idx_including_cls = raveled_desc_idx + 1
    curr_similarities = similarities[0, 0, reveled_desc_idx_including_cls, 1:]
    curr_similarities = curr_similarities.reshape(num_patches_b)
   # print(curr_similarities.shape)
    #print("min", min(curr_similarities[0]))
    #print("max", max(curr_similarities[0]))
    #print(patch_size)

    # get and draw most similar points
    sims, idxs = torch.topk(curr_similarities.flatten(), num_sim_patches)
    for idx, sim in zip(idxs, sims):
        y_descs_coor, x_descs_coor = idx // num_patches_b[1], idx % num_patches_b[1]
        center = ((x_descs_coor - 1) * stride + stride + patch_size // 2 - .5,
                    (y_descs_coor - 1) * stride + stride + patch_size // 2 - .5)
        #print("similar points",center)

if __name__ == '__main__':
    import numpy as np

    # Define the coordinates of the box
    x_start, x_end = 205, 215
    y_start, y_end = 140, 160

    pixel_array = np.empty((y_end - y_start + 1, x_end - x_start + 1), dtype=object)

    for y in range(y_start, y_end + 1):
        for x in range(x_start, x_end + 1):
            pixel_array[y - y_start, x - x_start] = (x, y)
    
    start_time = time.time()


    with torch.no_grad():
        shadow_sim_iter(r'datasets/export_3/trainA/9_rgb.png', pixel_array.flatten())

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time} seconds")
