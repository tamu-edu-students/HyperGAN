import torchvision.transforms as transforms
import numpy as np
import io
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from sklearn.metrics import mean_squared_error
from skimage.filters import threshold_otsu
from .eval_metrics import calculate_rmse, SSIM
from ..processor import Processor


def spectral_plot(
    gt_band_values, orig_band_values, rec_band_values, associated_pixel_color
):

    plt.clf()
    band_numbers = np.arange(1, len(gt_band_values) + 1)
    plt.plot(band_numbers, gt_band_values, label="Ground Truth")
    plt.plot(band_numbers, orig_band_values, label="Original Shadowed Image")
    plt.plot(band_numbers, rec_band_values, label="Reconstructed Image")
    plt.xlabel("Band Number")
    plt.ylabel("Reflectance Value")
    plt.title("Spectral Curves for Pixel: {}".format(associated_pixel_color))
    plt.legend()

    RMSE_orig = mean_squared_error(gt_band_values, orig_band_values)
    RMSE_rec = mean_squared_error(gt_band_values, rec_band_values)

    # plt.text(20, 4, 'RMSE: GT and Original' + f"{RMSE_orig:.3f}", fontsize=8, color='red', ha='center')
    # plt.text(20, 8, 'RMSE: GT and Reconstructed' + f"{RMSE_rec:.3f}", fontsize=8, color='blue', ha='center')

    buffer = io.BytesIO()

    # Save the Matplotlib plot to the BytesIO buffer as a PIL image
    plt.savefig(buffer, format="png")
    buffer.seek(0)  # Move the buffer cursor to the beginning

    # Open the PIL image from the BytesIO buffer
    return Image.open(buffer)


def highlight_selector(shadowed, deshadowed):

    to_gray = transforms.Grayscale(num_output_channels=1)
    im_f = to_gray(deshadowed)
    im_s = to_gray(shadowed)
    diff = np.asarray(im_f, dtype="float32") - np.asarray(im_s, dtype="float32")
    L = threshold_otsu(diff)
    mask = np.float32(diff >= L)

    print(max(mask), "amd", min(mask))


def hyper_to_gray():
    pass

def hyper_measures_eval(gt_hsi, resolved_hsi, rec_indxs):
    
    gt_spectra = []
    resolved_spectra = []

    p = Processor(hsi_data=gt_hsi)
    p2 = Processor(hsi_data=resolved_hsi)
    im1 = p.genFalseRGB(convertPIL=True)
    im2 = p2.genFalseRGB(convertPIL=True)
    
    box_size=4

    assert gt_hsi.shape == resolved_hsi.shape, "images are not the same size" 

    for x,y in rec_indxs:

        gt_spectra.append(gt_hsi[x, y, :])
        resolved_spectra.append(resolved_hsi[x, y, :])

    gt_avg_spectra = np.mean(gt_spectra, axis=0)
    resolved_avg_spectra = np.mean(resolved_spectra, axis=0)

    x = np.arange(51)
    plt.plot(x, gt_avg_spectra, label='gt')
    plt.plot(x, resolved_avg_spectra, label='resolved')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Plot of Two Arrays')
    plt.ylim(0,270)
    plt.legend()
    plt.grid(True)
    plt.show()

    rmse = calculate_rmse(gt_avg_spectra, resolved_avg_spectra)
    ssim = SSIM(gt_avg_spectra, resolved_avg_spectra)

    return rmse, ssim