import numpy as np
import rasterio
from spectral import msam
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
import numpy as np

def calculate_rmse(image1, image2):
    # Convert images to numpy arrays
    array1 = np.array(image1)
    array2 = np.array(image2)

    # Flatten the arrays
    flat_array1 = array1.flatten()
    flat_array2 = array2.flatten()

    # Calculate the mean squared error
    mse = mean_squared_error(flat_array1, flat_array2)

    # Calculate the root mean squared error (RMSE)
    rmse = np.sqrt(mse)

    return rmse

def PSNR(band_1, band_2):
    """Peak SNR calculation"""
    mse = mean_squared_error(band_1.ravel(), band_2.ravel())
    max_val = max(np.maximum(band_1.ravel(), band_2.ravel()))
    psnr = 10 * np.log10((max_val ** 2) / mse)
    return psnr

def SSIM(band_1, band_2):
    """Structural Similarity Index Measure"""
    min_val = min(np.minimum(band_1.ravel(), band_2.ravel()))
    max_val = max(np.maximum(band_1.ravel(), band_2.ravel()))
    return ssim(band_1, band_2, data_range=max_val-min_val)

def SAM():
    None


def covariance(band_1, band_2):
    """Covariance between bands"""
    return np.cov(band_1, band_2)

def correlation(band_1, band_2):
    """Correlation between bands"""
    return np.corrcoef(band_1, band_2)[0, 1]

def normalize(data):
    """Normalizes numpy arrays into scale 0.0 - 1.0"""
    array_min = data.min()
    array_max = data.max()

    return ((data - array_min) / (array_max - array_min))

