import time
from options.test_options import TestOptions 
from util import util
from models import networks, create_model
from models import create_model
import pylib as py
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image
import tqdm
import threading
from flask import Flask, render_template, Response
from dataset.maskshadow_dataset import MaskImageDataset
import queue
import torchvision.transforms as transforms
from io import BytesIO

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
            visuals = self.model.get_visuals(self.iters) #get visuals and convert to bytes
            visual_bytes = BytesIO()
            visuals.save(visual_bytes, format='JPEG')

            self.image_queue.put(visual_bytes.getvalue()) #place processed image in queue

            if self.iters == 200:
                break

app = Flask(__name__)

class WebOutput:
    def __init__(self, image_generator):
        self.image_generator = image_generator

    def run(self):
        @app.route('/')
        def index():
            return render_template('index.html')

        def generate():
            while True:
                generated_image = self.image_generator.image_queue.get()

                # Check for the sentinel value (None) to handle the end of image production
                if generated_image is None:
                    break

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + generated_image + b'\r\n\r\n')
                time.sleep(0.1)

        @app.route('/video_feed')
        def video_feed():
            return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

        app.run(debug=True, threaded=True)

if __name__ == '__main__':
    image_generator = Test() #initialize Test class
 
    web_output = WebOutput(image_generator) #initialize web output with generator output

    # Create and start the image generation thread
    image_thread = threading.Thread(target=image_generator.test_sequence) 
    image_thread.daemon = True
    image_thread.start()

    # Start the web output in the main thread
    web_output.run()




# if __name__ == '__main__':
    
#     opt = TestOptions().parse()

#     transforms_ = [#transforms.Resize((opt.size, opt.size), Image.BICUBIC),
#     transforms.Resize(int(opt.crop_size * 1.12), Image.Resampling.BICUBIC),
#     transforms.RandomCrop(opt.crop_size),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
#     dataloader = DataLoader(MaskImageDataset(opt.datasets_dir, opt.dataroot, transforms_=transforms_, unaligned=True, mode='test'),
#                 batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
    
#     model = create_model(opt)
#     model.data_length = len(dataloader.dataset)
#     model.setup(opt)

    
#     to_pil = transforms.ToPILImage()
#     iters = 0
#     py.mkdir(model.output_dir)
#     py.mkdir(model.sample_dir)

#     for i, batch in tqdm.tqdm(enumerate(dataloader), desc='Test Loop', total=len(dataloader.dataset)):
        
#         model.set_input(batch)
#         iters += 1
        
#         model.forward()
#         model.get_visuals(iters)

#         if iters == 200:
#             break
#     #model.expand_dataset()



