from PIL import Image
import numpy as np
from hyperspectral import processor
from hyperspectral.util.eval_metrics import calculate_rmse, SSIM
import tifffile

def merge_images(input_paths, input_paths2, input_paths3, input_paths4, output_path):
    images = [Image.open(path) for path in input_paths]
    images2 = [Image.open(path) for path in input_paths2]
    images3 = input_paths3
    images4 = [Image.open(path) for path in input_paths4]
    
    # Get the dimensions of the images
    width, height = images[0].size
    
    # Create a new image with width enough for all images in each row
    result_width = width * len(images)
    result_height = height * 4  # Two rows
    result = Image.new("RGB", (result_width, result_height))
    
    # Paste images in the first row
    for i, img in enumerate(images):
        img = img.resize((290,275))
        result.paste(img, (i * width, 0))
    
    # Paste images in the second row
    for i, img in enumerate(images2):
        img = img.resize((290,275))
        result.paste(img, (i * width, height))

    for i, img in enumerate(images3):
        img = img.resize((290,275))
        result.paste(img, (i * width, height*2))

    for i, img in enumerate(images4):
        img = img.resize((290,275))
        result.paste(img, (i * width, height*3))
    
    # Save the result
    result.save(output_path)



if __name__ == '__main__':

    # Example paths
    with tifffile.TiffFile('datasets/shadow_masks/resolved_20.tiff') as tif:
        image = tif.asarray()  # Convert the TIFF image to a numpy array
    p = processor.Processor(hsi_data=image)
    img1 = p.genFalseRGB(convertPIL=True)

    with tifffile.TiffFile('datasets/shadow_masks/resolved_30.tiff') as tif:
        image = tif.asarray()  # Convert the TIFF image to a numpy array
    p1 = processor.Processor(hsi_data=image)
    img2 = p1.genFalseRGB(convertPIL=True)

    with tifffile.TiffFile('datasets/shadow_masks/resolved_39.tiff') as tif:
        image = tif.asarray()  # Convert the TIFF image to a numpy array
    p2 = processor.Processor(hsi_data=image)
    img3 = p2.genFalseRGB(convertPIL=True)

    with tifffile.TiffFile('datasets/shadow_masks/resolved_47.tiff') as tif:
        image = tif.asarray()  # Convert the TIFF image to a numpy array
    p3 = processor.Processor(hsi_data=image)
    img4 = p3.genFalseRGB(convertPIL=True)

    with tifffile.TiffFile('datasets/shadow_masks/resolved_51.tiff') as tif:
        image = tif.asarray()  # Convert the TIFF image to a numpy array
    p4 = processor.Processor(hsi_data=image)
    img5 = p4.genFalseRGB(convertPIL=True)

    with tifffile.TiffFile('datasets/shadow_masks/resolved_58.tiff') as tif:
        image = tif.asarray()  # Convert the TIFF image to a numpy array
    p5 = processor.Processor(hsi_data=image)
    img6 = p5.genFalseRGB(convertPIL=True)

    with tifffile.TiffFile('datasets/shadow_masks/resolved_64.tiff') as tif:
        image = tif.asarray()  # Convert the TIFF image to a numpy array
    p6 = processor.Processor(hsi_data=image)
    img7 = p6.genFalseRGB(convertPIL=True)


    input_paths = ["datasets/eval_export/testA/session_000_022_snapshot_view.tiff", "datasets/eval_export/testA/session_000_030_snapshot_view.tiff", "datasets/eval_export/testA/session_000_039_snapshot_view.tiff",
                "datasets/eval_export/testA/session_000_047_snapshot_view.tiff", "datasets/eval_export/testA/session_000_051_snapshot_view.tiff", "datasets/eval_export/testA/session_000_058_snapshot_view.tiff",
                "datasets/eval_export/testA/session_000_064_snapshot_view.tiff"]
    input_paths2 = ["datasets/eval_export/testB/session_000_019_snapshot_view.tiff", "datasets/eval_export/testB/session_000_024_snapshot_view.tiff",
                    "datasets/eval_export/testB/session_000_035_snapshot_view.tiff", "datasets/eval_export/testB/session_000_043_snapshot_view.tiff",
                    "datasets/eval_export/testB/session_000_050_snapshot_view.tiff", "datasets/eval_export/testB/session_000_055_snapshot_view.tiff",
                    "datasets/eval_export/testB/session_000_060_snapshot_view.tiff"]
    
    input_paths3 = [img1, img2, img3, img4, img5, img6, img7]
    input_paths4 = ['output/ctf_pipeline/fine/img-2.png', 'output/ctf_pipeline/fine/img-4.png', 'output/ctf_pipeline/fine/img-6.png', 'output/ctf_pipeline/fine/img-9.png',
                    'output/ctf_pipeline/fine/img-11.png', 'output/ctf_pipeline/fine/img-13.png', "output/ctf_pipeline/fine/img-15.png"]

    output_path = "datasets/result.jpg"

    # Merge images
    merge_images(input_paths, input_paths2, input_paths3, input_paths4, output_path)


    p7 = processor.Processor()
    p7.prepare_data('datasets/ctf_pipeline/testA/39.tiff')
    p7.hsi_data = p7.hyperCrop2D(p7.hsi_data, 256,256)


    p8 = processor.Processor()
    p8.prepare_data('datasets/ctf_pipeline/ref/39.tiff')
    print("dimension", p8.hsi_data.shape)
    p8.hsi_data = p8.hyperCrop2D(p8.hsi_data, 256,256)
    print("dimension", p8.hsi_data.shape)

    with tifffile.TiffFile('datasets/shadow_masks/resolved_39_orig.tiff') as tif:
        image9 = tif.asarray()  # Convert the TIFF image to a numpy array
    p9 = processor.Processor(hsi_data=image9)
    p9.genFalseRGB
    #p9.hsi_data = p9.genArray()
    p9.hsi_data = p9.hyperCrop2D(p9.hsi_data, 256,256)

    with tifffile.TiffFile('datasets/shadow_masks/resolved_39.tiff') as tif:
        image10 = tif.asarray()  # Convert the TIFF image to a numpy array
    p10 = processor.Processor(hsi_data=image10)
    print("dimfension", p10.hsi_data.shape)
    #p10.hsi_data = p10.genArray()
    
    p10.hsi_data = p10.hyperCrop2D(p10.hsi_data, 256,256)
    print("dimension", p10.hsi_data.shape)

    print(" RMSE orig to gt ", SSIM(p8.hsi_data, p7.hsi_data))
    print(" RMSE orig to ctf1 ", SSIM(p8.hsi_data, p9.hsi_data)) 
    print(" RMSE orig to ctf2 ", SSIM(p8.hsi_data, p10.hsi_data))

    print(np.ptp(p9.hsi_data, axis=1))


    images = [Image.open("datasets/eval_export/testA/session_000_039_snapshot_view.tiff"),
              Image.open("datasets/eval_export/testB/session_000_035_snapshot_view.tiff"), 
              Image.open("output/ctf_pipeline/coarse/img-6.png"),
              p9.genFalseRGB(convertPIL=True),
              Image.open("output/ctf_pipeline/fine_only/img-6.png"),
              p10.genFalseRGB(convertPIL=True),
              Image.open("output/ctf_pipeline/fine/img-6.png"),
              ]


    width, height = images[0].size
    
    # Create a new image with width enough for all images in each row
    result_width = width * len(images)
    result_height = height   # Two rows
    result = Image.new("RGB", (result_width, result_height))
    
    # Paste images in the first row
    for i, img in enumerate(images):
        img = img.resize((290,275))
        result.paste(img, (i * width, 0))

    
    result.save("datasets/ablation.png")
