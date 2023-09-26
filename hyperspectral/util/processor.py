import numpy as np
import spectral as spy
import scipy.io as sio
import rasterio as rio
from PIL import Image
import matplotlib.pyplot as plt
import cv2

class Processor:
    def __init__(self, hsi_data=None) -> None:
        self.hsi_data = hsi_data

    def prepare_data(self, img_path):
        if img_path[-3:] == 'mat':
            img_mat = sio.loadmat(img_path)
            img_keys = img_mat.keys()
            img_key = [k for k in img_keys if k != '__version__' and k != '__header__' and k != '__globals__']
            return img_mat.get(img_key[0]).astype('float64')
        if img_path.find('tif') != -1:
            # Open the TIFF file using rasterio
            with rio.open(img_path) as src:
            # Read the data as a NumPy array
                self.hsi_data = src.read()
            # Access the shape of the data cube (bands, rows, columns)
            bands, rows, cols = self.hsi_data.shape
            print(bands, " ", rows, " ", cols)
            return self.hsi_data
        
        # if img_path.find('tif') != -1:
            
        #     hyperspectral_image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        #     # Reshape the image into a 3D NumPy array
        #     height, width, num_bands = hyperspectral_image.shape
        #     hsi_data = hyperspectral_image.reshape(height, width, -1)
        #     return hsi_data
        
        # if img_path.find('tif') != -1:
            
        #     hyperspectral_image = Image.open(img_path)

        #     # Convert the image to a NumPy array
        #     hsi_data = np.array(hyperspectral_image)
        #     return hsi_data
        
        else:
            try:
                img = spy.open_image(img_path).load()
                a = spy.principal_components()
                a.transform()
            except IOError:
                print("IOError: File format not supported")

    def convertMat(self, data):
        return sio.savemat(file_name='sundarbans_bands.mat', mdict={"data": data})
    

    def display_band(self, hsi_data: np.array, band_num: int):
        fig, ax = plt.subplots()

        # Display the grayscale image using imshow
        if band_num >= len(hsi_data) or band_num <0:
            raise IOError("Band number out of range")
        print("Showing band: ", band_num)
        intensity_array = hsi_data[band_num]
        ax.imshow(intensity_array, cmap='gray')

        # Add a colorbar for reference
        cbar = plt.colorbar(ax.imshow(intensity_array, cmap='gray'), ax=ax)

        # Set titles, labels, or other properties
        ax.set_title('Grayscale Intensity Image')
        cbar.set_label('Intensity')

        # Show the plot
        plt.show()
    
    def genFalseRGB(self, band_red: int, band_green:int, band_blue:int, hsi_data):
        
        bands, height, width = hsi_data.shape
        # image = Image.new("RGB", (width, height), "white")
        # # Get the pixel access object
        # pixels = image.load()

        red_band_norm = self.normalize_band(hsi_data[band_red])
        green_band_norm = self.normalize_band(hsi_data[band_green])
        blue_band_norm = self.normalize_band(hsi_data[band_blue])
        rgb_image = np.dstack((red_band_norm, green_band_norm, blue_band_norm))
        plt.imshow(rgb_image)
        plt.title('False RGB Image')
        plt.axis('off')
        plt.show()
        # for x in range(width):
        #     print(len(hsi_data[band_red][x]))
        #     for y in range(height):
                
        #         pixels[x, y] = (hsi_data[band_red][y][x], hsi_data[band_green][y][x], hsi_data[band_blue][y][x])  # Blue color in RGB
        # image.show()
    
    def normalize_band(self, band):
        return ((band - band.min()) / (band.max() - band.min()) * 255).astype(np.uint8)
