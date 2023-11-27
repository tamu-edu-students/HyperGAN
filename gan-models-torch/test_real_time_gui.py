import time
from options.test_options import TestOptions 
from util import util
from models import networks, create_model
from models import create_model
import pylib as py
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image, ImageTk
import tqdm
import threading
from flask import Flask, render_template, Response
from dataset.maskshadow_dataset import MaskImageDataset
import queue
import torchvision.transforms as transforms
from io import BytesIO
import tkinter as tk
import cv2
import warnings
import rasterio

class Test():

    def __init__(self) -> None:
        opt = TestOptions().parse() # parse options

        transforms_ = [transforms.Resize((opt.crop_size, opt.crop_size), Image.BICUBIC), #transforms function
        #transforms.Resize(int(opt.crop_size * 1.12), Image.Resampling.BICUBIC),
        transforms.ToTensor(), #convert image to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))] # standardize data
        self.dataloader = DataLoader(MaskImageDataset(opt.datasets_dir, opt.dataroot, transforms_=transforms_, unaligned=True, mode='test'),
                    batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu) #initialize dataloader with correct parameters
        
        self.model = create_model(opt) #create model using parsed model name
        self.model.data_length = len(self.dataloader.dataset) 
        self.model.setup(opt) # setup model

        
        self.iters = 0
        py.mkdir(self.model.output_dir) #make directories for model and sample output
        py.mkdir(self.model.sample_dir)

        self.image_queue = queue.Queue()

    def test_sequence(self):
        
        for i, batch in tqdm.tqdm(enumerate(self.dataloader), desc='Test Loop', total=len(self.dataloader.dataset)): #iterating through each data element
            
            self.model.set_input(batch) #setting input
            self.iters += 1
            self.model.forward() #run forward pass
            self.model.get_visuals(self.iters) #get visuals and convert to bytes
            
            # visual_bytes = BytesIO()
            # visuals.save(visual_bytes, format='JPEG')

            self.image_queue.put(np.array(self.model.output_image)) #place processed image in queue

            if self.iters == 200:
                break


class VideoCreator:
    def __init__(self, image_generator):
        self.image_generator = image_generator

    def create_video(self):
        # Video properties
        frame_width = 1440
        frame_height = 480
        out = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

        while True:
            with buffer_lock:

                frame = self.image_generator.image_queue.get()

            if frame is None:
                break  # End of image generation

            # Your video processing logic here
            # For simplicity, let's resize the frame
            frame = cv2.resize(frame, (frame_width, frame_height))

            # Write frame to video
            out.write(frame)

            cv2.imshow('Real-time Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        out.release()
        cv2.destroyAllWindows()

warnings.filterwarnings("ignore", message="Using a target size .* that is different to the input size .*")
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
# Global variables
buffer_lock = threading.Lock()
image_buffer = queue.Queue()

# Create instances of classes
image_generator = Test()
video_creator = VideoCreator(image_generator)

# Create threads
image_generation_thread = threading.Thread(target=image_generator.test_sequence)
video_creation_thread = threading.Thread(target=video_creator.create_video)

# Start threads
image_generation_thread.start()
video_creation_thread.start()

# Wait for threads to finish
image_generation_thread.join()
video_creation_thread.join()