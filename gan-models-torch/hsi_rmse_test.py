import time
from options.test_options import TestOptions 
from models import networks, create_model
from models import create_model
import pylib as py
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
import tqdm
from dataset.hyperspectral_dataset import HyperspectralImageDataset
import torchvision.transforms as transforms
from hyperspectral.util.eval_metrics import calculate_rmse, SSIM
from hyperspectral import processor
from hyperspectral.vit_reconstruction import fine_removal
import warnings
import rasterio
import tifffile

if __name__ == '__main__':
    warnings.filterwarnings("ignore", message="Using a target size .* that is different to the input size .*")
    warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
    opt = TestOptions().parse()

    mean_values = np.array([0.5] * opt.input_nc)  # input_nc channels with a mean of 0.5
    std_values = np.array([0.5] * opt.input_nc)   # input_nc channels with a standard deviation of 0.5

    transforms_ = [
    transforms.ToTensor(),
    transforms.Normalize(mean_values.tolist(), std_values.tolist())
    ]
    dataloader = DataLoader(HyperspectralImageDataset(opt.datasets_dir, opt.dataroot, True, transforms_=transforms_, unaligned=True, mode='test'),
                batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
    
    model = create_model(opt)
    model.data_length = len(dataloader.dataset)
    model.setup(opt)

    ref_dir = py.join(opt.datasets_dir, opt.dataroot, 'ref/*.*')
    mask_dir = py.join(opt.datasets_dir, opt.dataroot, 'mask/')
    coarse_dir = py.join(model.output_dir, 'coarse/')
    fine_dir = py.join(model.output_dir, 'fine_only/')

    py.mkdir(model.output_dir)
    py.mkdir(model.sample_dir)
    py.mkdir(coarse_dir)
    py.mkdir(fine_dir)

    iters = 0
    rec_ssim = []
    res_hyper = []

    for i, batch in tqdm.tqdm(enumerate(dataloader), desc='Test Loop', total=len(dataloader.dataset)):
        
        iters+=1
        model.set_input(batch)
        model.forward()
        pil_real_A, pil_fake_B, spec_real_A, spec_fake_B = model.get_visuals(iters)

        # model.save_mask()

        alternate_mask = Image.open(py.join(mask_dir, '{}.png'.format(i+1)))
        fine_result = fine_removal(alternate_mask, spec_real_A, spec_real_A)

        p = processor.Processor(hsi_data=fine_result)
  
        ##pil_fake_B.save(py.join(coarse_dir, "img-{}.png".format(i)))
        p.genFalseRGB(convertPIL=True).save(py.join(fine_dir, "img-{}.png".format(i)))
        #p.prepare_data(r'datasets/hsi_rmse/ref/session_000_001k_044_snapshot_cube.tiff')
        #real_B = p.genFalseRGB(convertPIL=True)
        # rec_ssim.append([SSIM(spec_real_A, p.hsi_data), SSIM(spec_fake_B, p.hsi_data)])
        ##tifffile.imwrite(py.join(coarse_dir, "img-{}.tiff".format(i)), spec_fake_B)
        tifffile.imwrite(py.join(fine_dir, "img-{}.tiff".format(i)), fine_result)


    
    # for i, val in enumerate(rec_ssim):
    #     print("Reconstruction for Image: ", i+1)
    #     print("SSIM: GT to Original Image: ", val[0])
    #     print("SSIM: GT to Enhanced Image: ", val[1])
    #     print()