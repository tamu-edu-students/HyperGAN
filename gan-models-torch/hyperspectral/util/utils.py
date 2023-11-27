import torchvision.transforms as transforms
import numpy as np
import io
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from sklearn.metrics import mean_squared_error
from skimage.filters import threshold_otsu


def spectral_plot(gt_band_values, orig_band_values, rec_band_values, associated_pixel_color):

    plt.clf()
    band_numbers = np.arange(1, len(gt_band_values)+1)
    plt.plot(band_numbers, gt_band_values,  label='Ground Truth')
    plt.plot(band_numbers, orig_band_values, label='Original Shadowed Image')
    plt.plot(band_numbers, rec_band_values, label='Reconstructed Image')
    plt.xlabel('Band Number')
    plt.ylabel('Reflectance Value')
    plt.title('Spectral Curves for Pixel: {}'.format(associated_pixel_color))
    plt.legend()


    RMSE_orig = mean_squared_error(gt_band_values, orig_band_values)
    RMSE_rec = mean_squared_error(gt_band_values, rec_band_values)

   # plt.text(20, 4, 'RMSE: GT and Original' + f"{RMSE_orig:.3f}", fontsize=8, color='red', ha='center')
   # plt.text(20, 8, 'RMSE: GT and Reconstructed' + f"{RMSE_rec:.3f}", fontsize=8, color='blue', ha='center')

    buffer = io.BytesIO()

    # Save the Matplotlib plot to the BytesIO buffer as a PIL image
    plt.savefig(buffer, format='png')
    buffer.seek(0)  # Move the buffer cursor to the beginning

    # Open the PIL image from the BytesIO buffer
    return Image.open(buffer)

def highlight_selector(shadowed, deshadowed):
    
    to_gray = transforms.Grayscale(num_output_channels=1)
    im_f = to_gray(deshadowed)
    im_s = to_gray(shadowed)
    diff = (np.asarray(im_f, dtype='float32')- np.asarray(im_s, dtype='float32'))
    L = threshold_otsu(diff)
    mask = np.float32(diff >= L)

    print(max(mask), "amd", min(mask))

def hyper_to_gray():
    pass