import numpy as np
import rasterio
from spectral import msam
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim

def RMSE(band_1, band_2):
    mse = mean_squared_error(band_1.ravel(), band_2.ravel())
    return np.sqrt(mse)

def PSNR(band_1, band_2):
    mse = mean_squared_error(band_1.ravel(), band_2.ravel())
    max_val = max(np.maximum(band_1.ravel(), band_2.ravel()))
    psnr = 10 * np.log10((max_val ** 2) / mse)
    return psnr

def SSIM(band_1, band_2):
    min_val = min(np.minimum(band_1.ravel(), band_2.ravel()))
    max_val = max(np.maximum(band_1.ravel(), band_2.ravel()))
    return ssim(band_1, band_2, data_range=max_val-min_val)

def SAM():
    None


def covariance(band_1, band_2):
    return np.cov(band_1, band_2)

def correlation(band_1, band_2):
    return np.corrcoef(band_1, band_2)[0, 1]

def normalize(data):
    """Normalizes numpy arrays into scale 0.0 - 1.0"""
    array_min = data.min()
    array_max = data.max()

    return ((data - array_min) / (array_max - array_min))

