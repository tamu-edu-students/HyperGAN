import numpy as np
import spectral as spy
import spectral.io.envi as envi
import scipy.io as sio
from scipy.interpolate import CubicHermiteSpline
import io
from scipy.ndimage import zoom
import rasterio as rio
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import cuvis
import time
import tifffile

class Processor:
    """
    This class serves functionality in regards to hyperspectral image processing through preparing data for tensor generation
    This class also provides provides additional functions such as a False RGB, array generation and band display
    """

    def __init__(self, hsi_data=None, user_settings=None) -> None:
        self.hsi_data = hsi_data
        self.user_settings = user_settings
        self.rows = 0
        self.cols = 0
        self.bands = 0

    def prepare_data(self, img_path):
        if img_path[-3:] == "mat":  # processing for matlab file
            img_mat = sio.loadmat(img_path)
            img_keys = img_mat.keys()
            img_key = [
                k
                for k in img_keys
                if k != "__version__" and k != "__header__" and k != "__globals__"
            ]
            self.hsi_data = img_mat.get(img_key[0]).astype("float64")

        if img_path.find("tif") != -1:
            # Open the TIFF file using rasterio
            with rio.open(img_path) as src:
                # Read the data as a NumPy array
                self.hsi_data = src.read()

            self.hsi_data = self.genArray()
            self.hsi_data = self.hyperCrop2D(self.hsi_data, 256,256)
            #print(self.hsi_data.shape)
            self.bands, self.rows, self.cols = self.hsi_data.shape

        if img_path.find("cu3") != -1:  # open file if in cubert format

            mesu = cuvis.Measurement(img_path)  # create measurement
            self.hsi_data = mesu.Data.pop("cube", None)  # pop cube
            self.rows, self.cols, self.bands = (
                self.hsi_data.array.shape
            )  # set dimensions
            print("bands: ", self.bands, " rows: ", self.rows, " cols: ", self.cols)

        if img_path.find("hdr") != -1:

            img_path_data = img_path[:-4]
            img = envi.open(
                img_path, img_path_data
            )  # use envi for hdr files with metadata
            arr = img.load()
            print(arr.info())
            self.hsi_data = arr

        if img_path.find("png") != -1:

            image = Image.open(img_path)
            #self.hsi_data = self.rgb_to_hyper(17, image, 'cubic')
            self.hsi_data = self.genArray()
            self.hsi_data = self.hyperCrop(self.hsi_data, 256)
            self.bands, self.rows, self.cols = self.hsi_data.shape

        return self.hsi_data  # returns data
    
    def hyperCrop2D(self, arr, target_dimX, target_dimY):

        target_shape = (target_dimX, target_dimY)  # input crop sizes
        rec_array = np.empty(
            (target_dimX, target_dimY, arr.shape[2]), dtype=np.float32
        )  # empty 3D array
        arr_list = []

        for i in range(arr.shape[2]):  # iterating through dimensions for cropping
            twoD = arr[:, :, i]  # capturing 2D array at each dimension
            zoom_factors = (
                target_shape[0] / twoD.shape[0],
                target_shape[1] / twoD.shape[1],
            )  # creating zoom factor by comparing dimensions
            resized_array = zoom(twoD, zoom_factors)  # using scipy function for zoom
            arr_list.append(resized_array)

        return np.dstack(arr_list)  # stacking dimensions for output

    def convertMat(self, data):
        return sio.savemat(file_name="sundarbans_bands.mat", mdict={"data": data})

    def display_band(self, hsi_data: np.array, band_num: int):
        fig, ax = plt.subplots()

        # Display the grayscale image using imshow
        if band_num >= self.bands or band_num < 0:
            raise IOError("Band number out of range")
        print("Showing band: ", band_num)
        intensity_array = hsi_data[band_num]
        ax.imshow(intensity_array, cmap="gray")

        # Add a colorbar for reference
        cbar = plt.colorbar(ax.imshow(intensity_array, cmap="gray"), ax=ax)

        # Set titles, labels, or other properties
        ax.set_title("Grayscale Intensity Image")
        cbar.set_label("Intensity")

        # Show the plot
        plt.show()

    def genFalseRGB(
        self,
        band_red: int = 25,
        band_green: int = 12,
        band_blue: int = 3,
        visualize=False,
        convertPIL=False,
    ):

        bands, height, width = self.hsi_data.shape

        red_band_norm = self.normalize_band(
            self.hsi_data[:, :, band_red]
        )  # normalizing values of red band
        green_band_norm = self.normalize_band(
            self.hsi_data[:, :, band_green]
        )  # normalizing values of green band
        blue_band_norm = self.normalize_band(
            self.hsi_data[:, :, band_blue]
        )  # normalizing values of blue band
        rgb_image = np.dstack(
            (red_band_norm, green_band_norm, blue_band_norm)
        )  # stacking three dimensions for false RGB output
        rgb_image = np.uint8(rgb_image)

        plt.imshow(rgb_image) if visualize else None
        plt.show() if visualize else None

        if convertPIL:
            return Image.fromarray(rgb_image)  # convert to PIL image if desired
        else:
            return rgb_image

    def genArray(self):

        bands, height, width = (
            self.hsi_data.shape
        )  # capturing dimensions of measurement
        #self.hsi_data = np.transpose(self.hsi_data, axes=(0, 2, 1))
        arr_list = []
        for i in range(bands):
            arr_list.append(
                self.normalize_band(self.hsi_data[i, :, :])
            )  # creating numpy array after normalizing data

        return np.uint8(np.dstack(arr_list))

    def normalize_band(self, band):
        return ((band - band.min()) / (band.max() - band.min()) * 255).astype(
            np.uint8
        )  # normalizing data from band to uint8 format

#     def rgb_to_hyper(self, cube_rep, image, fit):
#         """
#         fit : (str) Can be repeat, linear, or cubic. Determines the interpolation when inflating image.
#         """

#         r, g, b = image.split()
        
#         colors = [np.array(r), np.array(g), np.array(b)]
       
#         arr_list = []

#         if fit == 'repeat':

#             for dim in colors:
#                 for j in range(cube_rep):
#                     arr_list.append(dim)
#             arr_list = np.stack(arr_list)
        
#         if fit == 'cubic':
#             arr_list = np.zeros((275, 290, 51))
#             dy_dx = np.array([0, 0])
#             start_time = time.time()
#             for i in range(colors[0].shape[0]):
#                 for j in range(colors[0].shape[1]):
#                     r_val = colors[0][i][j]
#                     g_val = colors[1][i][j]
#                     b_val = colors[2][i][j]

#                     x = np.array([1, 26, 51])
#                     y = np.array([b_val, g_val, r_val])

#                     cubic_spline_bg = CubicHermiteSpline(x[:2], y[:2], dy_dx)
#                     b_g_x = np.arange(x[0], x[1])
#                     b_g_y = cubic_spline_bg(b_g_x)

#                     cubic_spline_gr = CubicHermiteSpline(x[1:], y[1:], dy_dx)
#                     g_r_x = np.arange(x[1], x[2]+1)
#                     g_r_y = cubic_spline_gr(g_r_x)
                    

#                     arr_list[i][j] = np.append(b_g_y, g_r_y)

#                     # if i == 200 and j == 146:
#                     #     print("red value", r_val)
#                     #     print("green value", g_val)
#                     #     print("blue value", b_val)

#                     #     plt.plot(arr_list[i][j], marker='o', linestyle='-')
#                     #     plt.title('Plot of Values')
#                     #     plt.xlabel('Index')
#                     #     plt.ylabel('Random Values')
#                     #     plt.grid(True)
#                     #     plt.show()                        

#             end_time = time.time()

# # Calculate the elapsed time
#             elapsed_time = end_time - start_time

#             # Print the result
#             print(f"Time spent in the loop: {elapsed_time} seconds")



#         return arr_list

with tifffile.TiffFile('datasets/shadow_masks/resolved_48_dino.tiff') as tif:
    image = tif.asarray()  # Convert the TIFF image to a numpy array
p = Processor(hsi_data=image)
# # p.prepare_data(r'datasets/export_2/trainA/session_000_001k_048_snapshot_ref.tiff')
# #print(p.hsi_data.shape)
p.genFalseRGB(visualize=True)
# cropped_region = p.hsi_data[10:75, 12:24, :]
# print(cropped_region.shape)
# average_hyper = np.mean(cropped_region, axis=(0, 1))
# print(average_hyper.shape)
# print("Average hyper Values:", average_hyper)
# # # # p.genFalseRGB(25,12,3,visualize=True)
# # # # # #plt.imshow(p.hsi_data[12, :, :], cmap='gray')
# # # # plt.show()
# print(p.genFalseRGB().shape)
# # # # p.prepare_data(r'/workspaces/HyperGAN/datasets/hsi/trainA/session_000_023_snapshot.cu3')
# # # # plt.imshow(p.hsi_data[:, :, 12], cmap='gray')
# # # # plt.show()

# # print(p.hsi_data.shape)
# # # # rgb = p.genFalseRGB(27, 14, 8, visualize=True)
