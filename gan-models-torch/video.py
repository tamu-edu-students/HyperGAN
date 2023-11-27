# import cv2
# import os

# image_folder = r'/workspaces/HyperGAN/output/sda/samples_testing'
# video_name = 'video.avi'


# images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
# frame = cv2.imread(os.path.join(image_folder, images[0]))
# height, width, layers = frame.shape

# video = cv2.VideoWriter(video_name, 0, 10, (width,height))

# for image in images:
#     video.write(cv2.imread(os.path.join(image_folder, image)))

# cv2.destroyAllWindows()
# video.release()

import threading
import queue
import time
from flask import Flask, render_template, Response
from PIL import Image
from io import BytesIO

app = Flask(__name__)
image_queue = queue.Queue()

def generate_random_color_image():
    # Generate a random RGB color
    color = (int(time.time() * 100) % 256, int(time.time() * 50) % 256, int(time.time() * 25) % 256)

    # Create a 100x100 image with the random color
    image = Image.new('RGB', (100, 100), color)

    # Convert the image to bytes
    image_bytes = BytesIO()
    image.save(image_bytes, format='JPEG')
    return image_bytes.getvalue()

def image_generator():
    while True:
        # Generate a random color image
        generated_image = generate_random_color_image()
        print("generating")

        # Put the generated image into the queue
        image_queue.put(generated_image)
        time.sleep(0.5)

def web_output():
    @app.route('/')
    def index():
        return render_template('index.html')

    def generate():
        while True:
            # Get the latest image from the queue
            generated_image = image_queue.get()

            # Yield the image data to the web page
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + generated_image + b'\r\n\r\n')
            time.sleep(0.1)

    @app.route('/video_feed')
    def video_feed():
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    app.run(debug=True, threaded=True)

if __name__ == '__main__':
    # Create and start the image generation thread
    image_thread = threading.Thread(target=image_generator)
    image_thread.daemon = True
    image_thread.start()

    # Start the web output in the main thread
    web_output()
