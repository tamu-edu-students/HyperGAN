from flask import Flask, render_template, Response
import cv2
import os
import time

app = Flask(__name__)

# OpenCV VideoCapture object (use 0 for default camera)
cap = cv2.VideoCapture(0)

def generate_frames():
     image_folder = r'/workspaces/HyperGAN/output/sda/samples_testing'
     for filename in sorted(os.listdir(image_folder)):
        if filename.endswith('.jpg'):
            image_path = os.path.join(image_folder, filename)
            with open(image_path, 'rb') as f:
                frame = f.read()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
                time.sleep(0.2)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
