import numpy as np
import spectral as spy
import spectral.io.envi as envi
import scipy.io as sio
from scipy.ndimage import zoom
import rasterio as rio
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import cuvis
from . import cuvis_tools

class Processor:
    def __init__(self, hsi_data=None, user_settings=None) -> None:
        self.hsi_data = hsi_data
        self.user_settings = user_settings
        self.rows = 0
        self.cols = 0
        self.bands = 0

    def prepare_data(self, img_path):
        if img_path[-3:] == 'mat':
            img_mat = sio.loadmat(img_path)
            img_keys = img_mat.keys()
            img_key = [k for k in img_keys if k != '__version__' and k != '__header__' and k != '__globals__']
            self.hsi_data = img_mat.get(img_key[0]).astype('float64')
        
        if img_path.find('tif') != -1:
            # Open the TIFF file using rasterio
            with rio.open(img_path) as src:
            # Read the data as a NumPy array
                self.hsi_data = src.read()
            
            self.hsi_data = self.genArray()
            self.hsi_data = self.hyperCrop(self.hsi_data, 256)
            self.bands, self.rows, self.cols = self.hsi_data.shape
           
            #print("bands: ", self.bands, " rows: ", self.rows, " cols: ", self.cols)
        
        if img_path.find('cu3') != -1:

            mesu = cuvis.Measurement(img_path)
            cuvis_tools.prettyprint_attributes(mesu)
            self.hsi_data = mesu.Data.pop("cube", None)
            self.rows, self.cols, self.bands = self.hsi_data.shape
            print("bands: ", self.bands, " rows: ", self.rows, " cols: ", self.cols)

        if img_path.find('hdr') != -1:

            # shape = (3, 290, 275)
            # self.hsi_data = np.random.rand(*shape)
            # Generate random noise in the range [0, 1]
            
            img_path_data = img_path[:-4]
            img = envi.open(img_path, img_path_data)
            arr = img.load()
            print(arr.info())
            self.hsi_data = arr
            

        return self.hsi_data

    def hyperCrop(self, arr, target_dim):

        target_shape = (target_dim, target_dim)
        rec_array = np.empty((target_dim, target_dim, arr.shape[2]), dtype=np.float32)
        arr_list = []

        for i in range(arr.shape[2]):
            twoD = arr[:,:,i]
            zoom_factors = (target_shape[0] / twoD.shape[0], target_shape[1] / twoD.shape[1])
            resized_array = zoom(twoD, zoom_factors)
            arr_list.append(resized_array)
        
        return np.dstack(arr_list)


    def convertMat(self, data):
        return sio.savemat(file_name='sundarbans_bands.mat', mdict={"data": data})
    
    def display_band(self, hsi_data: np.array, band_num: int):
        fig, ax = plt.subplots()

        # Display the grayscale image using imshow
        if band_num >= self.bands or band_num <0:
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


    def genFalseRGB(self, band_red: int = 25, band_green:int = 12, band_blue:int = 3, visualize=False, convertPIL=False):
        
        bands, height, width = self.hsi_data.shape

        red_band_norm = self.normalize_band(self.hsi_data[band_red, :, ])
        green_band_norm = self.normalize_band(self.hsi_data[band_green, :, :])
        blue_band_norm = self.normalize_band(self.hsi_data[band_blue, :, :])
        rgb_image = np.dstack((red_band_norm, green_band_norm, blue_band_norm))
        rgb_image = np.uint8(rgb_image)
        
        plt.imshow(rgb_image) if visualize else None
        plt.show() if visualize else None

        if convertPIL:
            return Image.fromarray(rgb_image)
        else:
            return rgb_image
        
    def genArray(self):
        
        bands, height, width = self.hsi_data.shape
        arr_list = []
        for i in range(bands):
            arr_list.append(self.normalize_band(self.hsi_data[i, :, :]))

        return np.uint8(np.dstack(arr_list))


    def normalize_band(self, band):
        return ((band - band.min()) / (band.max() - band.min()) * 255).astype(np.uint8)

p = Processor() 
p.prepare_data(r'/workspaces/HyperGAN/datasets/export/trainB/session_000_058_snapshot_ref.tiff')
# # # p.genFalseRGB(25,12,3,visualize=True)
# # # # #plt.imshow(p.hsi_data[12, :, :], cmap='gray')
# # # plt.show()

# # # p.prepare_data(r'/workspaces/HyperGAN/datasets/hsi/trainA/session_000_023_snapshot.cu3')
# # # plt.imshow(p.hsi_data[:, :, 12], cmap='gray')
# # # plt.show()

# print(p.hsi_data.shape)
# # # rgb = p.genFalseRGB(27, 14, 8, visualize=True)