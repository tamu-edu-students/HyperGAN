# from processor import Processor
# import eval_metrics
import os
import glob
import cuvis
from cuvis.cuvis_types import ProcessingMode
import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib import animation
import json
from multiprocessing import cpu_count, Process, Value, Array, Pool, TimeoutError
from processor import Processor
from PIL import Image
import tifffile
from scipy.interpolate import CubicHermiteSpline

# from cuvis.Export import EnviExporter


def reprocessMeasurement(
    userSettingsDir,
    measurementLoc,
    darkLoc,
    whiteLoc,
    distanceLoc,
    factoryDir,
    outDir,
    visualize=False,
    export=False,
):

    print("loading user settings...")
    settings = cuvis.General(userSettingsDir)
    settings.setLogLevel("info")

    print("loading measurement file...")
    mesu = cuvis.Measurement(measurementLoc)

    print("loading dark...")
    dark = cuvis.Measurement(darkLoc)
    print("loading white...")
    white = cuvis.Measurement(whiteLoc)
    print("loading dark...")
    distance = cuvis.Measurement(distanceLoc)

    print(
        "Data 1 {} t={}ms mode={}".format(
            mesu.Name,
            mesu.IntegrationTime,
            mesu.ProcessingMode,
        )
    )

    print("loading calibration and processing context (factory)...")
    calibration = cuvis.Calibration(base=factoryDir)
    processingContext = cuvis.ProcessingContext(calibration)

    print("set references...")
    processingContext.setReference(dark, "Dark")
    processingContext.setReference(white, "White")
    processingContext.setReference(distance, "Distance")

    modes = ["Reflectance"]

    procArgs = cuvis.CubertProcessingArgs()
    saveArgs = cuvis.CubertSaveArgs(AllowOverwrite=True)

    for mode in modes:

        procArgs.ProcessingMode = mode
        isCapable = processingContext.isCapable(mesu, procArgs)

        if isCapable:
            print("processing to mode {}...".format(mode))
            processingContext.setProcessingArgs(procArgs)
            mesu = processingContext.apply(mesu)
            cube = mesu.Data.pop("cube", None)
            if export:
                tiff_settings = cuvis.TiffExportSettings(ExportDir=outDir)
                tiffExporter = cuvis.TiffExporter(tiff_settings)
                print("saving ", outDir)
                tiffExporter.apply(mesu)
            if cube is None:
                raise Exception("Cube not found")

        else:
            print("Cannot process to {} mode!".format(mode))

    if visualize == True:
        # below is to visualize the image
        # print((cube.array.shape))
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(cube.array[:, :, 0], cmap=plt.cm.gray, animated=True)

        def animated_image(i):
            # cube.array[x, y, chn]
            # for chn in np.arange(cube.channels):
            data = cube.array[:, :, i]
            # x_norm = data*10
            x_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
            # im.set_array(x_norm*65535)
            # converts to 8 bit
            img8 = (x_norm * 255).astype("uint8")
            im.set_array(img8 * 256)

            plt.title(
                "Hyperspectral Image Layer {} of {} with wavelentgth {}".format(
                    i, mesu.Name, cube.wavelength[i]
                )
            )

        ani = animation.FuncAnimation(
            fig, animated_image, frames=cube.channels, interval=20, repeat=False
        )
        plt.show()

    # print("finished.")
    return cube, mesu


def loadMeasurement(userSettingsDir, measurementLoc, visualize=False):

    print("loading user settings...")
    settings = cuvis.General(userSettingsDir)
    settings.setLogLevel("info")

    print("loading measurement file...")
    mesu = cuvis.Measurement(measurementLoc)
    print(
        "Data 1 {} t={}ms mode={}".format(
            mesu.Name,
            mesu.IntegrationTime,
            mesu.ProcessingMode,
        )
    )

    if not isinstance(mesu.MeasurementFlags, list):
        mesu.MeasurementFlags = [mesu.MeasurementFlags]

    if len(mesu.MeasurementFlags) > 0:
        print("Flags")
        for flag in mesu.MeasurementFlags:
            print(" - {} ({})".format(flag, flag))

    assert mesu.ProcessingMode == "Raw", "This example requires Raw mode!"

    cube = mesu.Data.pop("cube", None)
    if cube is None:
        raise Exception("Cube not found")

    if visualize == True:
        # below is to visualize the image
        # print((cube.array.shape))
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(cube.array[:, :, 0], cmap=plt.cm.gray, animated=True)

        def animated_image(i):
            # cube.array[x, y, chn]
            # for chn in np.arange(cube.channels):
            data = cube.array[:, :, i]
            x_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
            im.set_array(x_norm * 50)

            plt.title(
                "Hyperspectral Image Layer {} of {} with wavelength {}".format(
                    i, mesu.Name, cube.wavelength[i]
                )
            )

        print(type(cube.array[5, 5, 5]))
        ani = animation.FuncAnimation(
            fig, animated_image, frames=cube.channels, interval=20, repeat=False
        )
        plt.show()

    # print("finished.")
    return cube


def extract_rgb(cube, red_layer=25, green_layer=12, blue_layer=3, visualize=False):

    red_img = cube.array[:, :, red_layer]
    green_img = cube.array[:, :, green_layer]
    blue_img = cube.array[:, :, blue_layer]
    data = np.stack([red_img, green_img, blue_img], axis=-1)
    # print(image.shape)
    # print(type(image))

    # convert to 8bit
    x_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
    image = (x_norm * 255).astype("uint8")
    if visualize:
        # pass
        plt.imshow(image)
        plt.show()
    image = Image.fromarray(image)
    return image


def prettyprint_attributes(mesu: cuvis.Measurement) -> None:
    for attribute in dir(mesu):
        if (
            type(getattr(mesu, attribute)).__name__
            not in ["builtin_function_or_method", "method"]
            and "__" not in attribute
        ):
            print(f"{attribute}:\t{getattr(mesu, attribute)}")


def exporter(file_type, domain):

    userSettingsDir = "./datasets/hsi/ultris5/"
    measurementLoc = "./datasets/eval_export/test{}/".format(domain)
    darkLoc = (
        "./datasets/eval_export/Calibration/dark__session_000_004_snapshot17098596920562928.cu3"
    )
    whiteLoc = "./datasets/eval_export/Calibration/white__session_000_012_snapshot17098601870622098.cu3"
    distanceLoc = "./datasets/eval_export/Calibration/distanceCalib__session_000_006_snapshot17098598097816240.cu3"
    factoryDir = "./datasets/hsi/ultris5/"
    outDir = "./datasets/eval/test{}/".format(domain)
    os.makedirs(outDir, exist_ok=True)

    cu3_files = glob.glob(os.path.join(measurementLoc, "*.cu3"))
    iter = 0

    for cu3_file in cu3_files:
        mesu = cuvis.Measurement(cu3_file)
        prettyprint_attributes(mesu)
        if file_type == "tiff":
            print(type(mesu.processing_mode), "yea")
            if mesu.processing_mode != ProcessingMode.Reflectance:
                break
                reprocessMeasurement(
                    userSettingsDir,
                    cu3_file,
                    darkLoc,
                    whiteLoc,
                    distanceLoc,
                    factoryDir,
                    os.path.join(outDir, "train{}".format(domain)),
                    visualize=False,
                    export=True,
                )
            else:
                tiff_settings = cuvis.TiffExportSettings(
                    export_dir=os.path.join(outDir, "train{}".format(domain))
                )
                tiffExporter = cuvis.TiffExporter(tiff_settings)
                print("saving ", cu3_file)
                tiffExporter.apply(mesu)
        if file_type == "png":
            p = Processor()
            print("file,", cu3_file)
            iter += 1
            hsi_data = p.prepare_data(cu3_file)
            image = extract_rgb(hsi_data)
            image.save(os.path.join(outDir, "{}_rgb.png".format(iter)))

def rgb_to_hyper(domain, fit=None, cube_rep=17):
    """
    fit : (str) Can be repeat, linear, or cubic. Determines the interpolation when inflating image.
    """
    measurementLoc = "./datasets/shadow_USR/train{}/".format(domain)
    outDir = "./datasets/transfer_experiment/train{}/".format(domain)

    png_files = glob.glob(os.path.join(measurementLoc, "*.jpg"))
    iter = 80

    for png_file in png_files:
        image = Image.open(png_file)
        image = image.resize((290,275))

        r, g, b = image.split()
        
        colors = [np.array(r), np.array(g), np.array(b)]
        print(colors[0].shape)
        arr_list = []

        if fit == 'repeat':

            for dim in colors:
                for j in range(cube_rep):
                    arr_list.append(dim)
            arr_list = np.stack(arr_list)
        
        if fit == 'cubic':
            
            arr_list = np.zeros((275, 290, 51))
            dy_dx = np.array([0, 0])
            start_time = time.time()
            for i in range(colors[0].shape[0]):
                for j in range(colors[0].shape[1]):
                    r_val = colors[0][i][j]
                    g_val = colors[1][i][j]
                    b_val = colors[2][i][j]

                    x = np.array([1, 26, 51])
                    y = np.array([b_val, g_val, r_val])

                    cubic_spline_bg = CubicHermiteSpline(x[:2], y[:2], dy_dx)
                    b_g_x = np.arange(x[0], x[1])
                    b_g_y = cubic_spline_bg(b_g_x)

                    cubic_spline_gr = CubicHermiteSpline(x[1:], y[1:], dy_dx)
                    g_r_x = np.arange(x[1], x[2]+1)
                    g_r_y = cubic_spline_gr(g_r_x)
                    arr_list[i][j] = np.append(b_g_y, g_r_y)


            end_time = time.time()

# Calculate the elapsed time
            elapsed_time = end_time - start_time
            print("final shape" , arr_list.shape)
            tifffile.imwrite(os.path.join(outDir, "{}_inflated.tiff".format(iter)), arr_list)
            # Print the result
            print(f"Time spent in the loop: {elapsed_time} seconds")
            print("iter ", iter, "path ", png_file)
            iter += 1
            print()





if __name__ == "__main__":

    exporter("tiff", "B")
    # rgb_to_hyper("B", 'cubic')
