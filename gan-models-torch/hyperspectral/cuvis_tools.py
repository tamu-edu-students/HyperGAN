#from processor import Processor
#import eval_metrics
import os
import glob
import cuvis
import matplotlib.pyplot as plt
import numpy as np
import time 
from matplotlib import animation
import json 
from multiprocessing import cpu_count, Process, Value, Array, Pool, TimeoutError
from processor import Processor
from PIL import Image
#from cuvis.Export import EnviExporter

def reprocessMeasurement(userSettingsDir,measurementLoc,darkLoc,whiteLoc,distanceLoc,factoryDir,outDir, visualize=False, export=False):    
    
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

    print("Data 1 {} t={}ms mode={}".format(mesu.Name,mesu.IntegrationTime,mesu.ProcessingMode,))

    print("loading calibration and processing context (factory)...")
    calibration = cuvis.Calibration(base=factoryDir)
    processingContext = cuvis.ProcessingContext(calibration)

    print("set references...")
    processingContext.setReference(dark, "Dark")
    processingContext.setReference(white, "White")
    processingContext.setReference(distance, "Distance")

    modes = [
             "Reflectance"
             ]

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
                tiff_settings = cuvis.TiffExportSettings(
                    ExportDir=outDir)
                tiffExporter = cuvis.TiffExporter(tiff_settings)
                print("saving ", outDir)
                tiffExporter.apply(mesu)
            if cube is None:
                raise Exception("Cube not found")

        else:
            print("Cannot process to {} mode!".format(mode))
            
            
    if visualize==True:
        # below is to visualize the image
        #print((cube.array.shape))
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        im = ax.imshow(cube.array[:,:, 0],cmap=plt.cm.gray, animated=True)

        def animated_image(i):
            # cube.array[x, y, chn]
            #for chn in np.arange(cube.channels):
            data = cube.array[:,:, i]
            #x_norm = data*10
            x_norm = (data-np.min(data))/(np.max(data)-np.min(data))
            #im.set_array(x_norm*65535)
            #converts to 8 bit 
            img8=(x_norm*255).astype('uint8')
            im.set_array(img8*256)

            plt.title("Hyperspectral Image Layer {} of {} with wavelentgth {}".format(i, mesu.Name, cube.wavelength[i]))

        ani = animation.FuncAnimation(fig, animated_image,frames=cube.channels, interval=20, repeat=False)
        plt.show()
    
    #print("finished.")
    return cube, mesu
    
def loadMeasurement(userSettingsDir, measurementLoc, visualize=False):

    print("loading user settings...")
    settings = cuvis.General(userSettingsDir)
    settings.setLogLevel("info")

    print("loading measurement file...")
    mesu = cuvis.Measurement(measurementLoc)
    print("Data 1 {} t={}ms mode={}".format(mesu.Name,mesu.IntegrationTime,mesu.ProcessingMode,))

    if not isinstance(mesu.MeasurementFlags, list):
        mesu.MeasurementFlags = [mesu.MeasurementFlags]

    if len(mesu.MeasurementFlags) > 0:
        print("Flags")
        for flag in mesu.MeasurementFlags:
            print(" - {} ({})".format(flag, flag)) 

    assert mesu.ProcessingMode == "Raw", \
        "This example requires Raw mode!"

    cube = mesu.Data.pop("cube", None)
    if cube is None:
        raise Exception("Cube not found")
   
    if visualize==True:
        # below is to visualize the image
        #print((cube.array.shape))
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        im = ax.imshow(cube.array[:,:, 0],cmap=plt.cm.gray, animated=True)

        def animated_image(i):
            # cube.array[x, y, chn]
            #for chn in np.arange(cube.channels):
            data = cube.array[:,:, i]
            x_norm = (data-np.min(data))/(np.max(data)-np.min(data))
            im.set_array(x_norm*50)

            plt.title("Hyperspectral Image Layer {} of {} with wavelentgth {}".format(i, mesu.Name, cube.wavelength[i]))

        print(type(cube.array[5,5, 5]))    
        ani = animation.FuncAnimation(fig, animated_image,frames=cube.channels, interval=20, repeat=False)
        plt.show()

    #print("finished.")
    return cube


def extract_rgb(cube, red_layer=25 , green_layer=12, blue_layer=3,  visualize=False):

    red_img=cube.array[:,:, red_layer]
    green_img=cube.array[:,:, green_layer]
    blue_img=cube.array[:,:, blue_layer]
    data=np.stack([red_img,green_img,blue_img], axis=-1)
    #print(image.shape)
    #print(type(image))

    #convert to 8bit
    x_norm = (data-np.min(data))/(np.max(data)-np.min(data))
    image=(x_norm*255).astype('uint8')
    if visualize:
        #pass
        plt.imshow(image)
        plt.show()
    image = Image.fromarray(image)
    return image  


def prettyprint_attributes(mesu: cuvis.Measurement) -> None:
    for attribute in dir(mesu):
        if type(getattr(mesu, attribute)).__name__ not in ['builtin_function_or_method', 'method'] and '__' not in attribute:
            print(f'{attribute}:\t{getattr(mesu, attribute)}')

def exporter(file_type, domain):
    
    userSettingsDir = "./datasets/real_time/ultris5/" 
    measurementLoc = "./datasets/real_time/test{}/".format(domain) 
    darkLoc = "./datasets/hsi/Calibration/dark__session_000_015_snapshot16976577013877078.cu3"
    whiteLoc = "./datasets/hsi/Calibration/white__session_000_017_snapshot16976577395328359.cu3"
    distanceLoc = "./datasets/hsi/Calibration/distanceCalib__session_000_010_snapshot16976559049021536.cu3"
    factoryDir = "./datasets/real_time/ultris5/"
    outDir = "./datasets/hsi_real_time_2/test{}/".format(domain)
    os.makedirs(outDir, exist_ok=True)
    

    cu3_files = glob.glob(os.path.join(measurementLoc, "*.cu3"))
    iter = 0

    for cu3_file in cu3_files:
        mesu = cuvis.Measurement(cu3_file)
        prettyprint_attributes(mesu)
        if file_type == "tiff":
            if mesu.ProcessingMode != "Reflectance":
                break
                reprocessMeasurement(userSettingsDir,cu3_file,darkLoc,whiteLoc,distanceLoc,factoryDir,os.path.join(outDir, "train{}".format(domain)), visualize=False, export=True)
            else:
                tiff_settings = cuvis.TiffExportSettings(
                    ExportDir=os.path.join(outDir, "train{}".format(domain)))
                tiffExporter = cuvis.TiffExporter(tiff_settings)
                print("saving ", cu3_file)
                tiffExporter.apply(mesu)
        if file_type == 'png':
            p = Processor()
            print("file,", cu3_file)
            iter+=1
            hsi_data = p.prepare_data(cu3_file)
            image = extract_rgb(hsi_data)
            image.save(os.path.join(outDir, '{}_rgb.png'.format(iter)))




if __name__ == "__main__":

    exporter("png", "A")
