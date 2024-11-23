from flask import Flask, render_template, request, redirect, url_for
import os
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load model and classes
model = load_model('traffic_sign_model.keras')

# Classes dictionary
classes = { 0:'Speed limit (20km/h)',1:'Speed limit (30km/h)',2:'Speed limit (50km/h)',3:'Speed limit (60km/h)',4:'Speed limit (70km/h)',
            5:'Speed limit (80km/h)',6:'End of speed limit (80km/h)',7:'Speed limit (100km/h)',8:'Speed limit (120km/h)',9:'No passing',
            10:'No passing veh over 3.5 tons',11:'Right-of-way at intersection',12:'Priority road',13:'Yield',14:'Stop',
            15:'No vehicles', 16:'Veh > 3.5 tons prohibited',17:'No entry',18:'General caution',19:'Dangerous curve left',
            20:'Dangerous curve right', 21:'Double curve',
            22:'Bumpy road', 23:'Slippery road', 24:'Road narrows on the right', 25:'Road work', 26:'Traffic signals', 27:'Pedestrians', 28:'Children crossing', 29:'Bicycles crossing',
            30:'Beware of ice/snow',31:'Wild animals crossing', 32:'End speed + passing limits', 33:'Turn right ahead',34:'Turn left ahead',
            35:'Ahead only',36:'Go straight or right', 37:'Go straight or left', 38:'Keep right',39:'Keep left',40:'Roundabout mandatory',41:'End of no passing',42:'End no passing veh > 3.5 tons' }

# Function to preprocess image
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((30, 30))  # Adjust to your input size
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    return image

# Homepage route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    # Save file
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Preprocess and predict
        image = preprocess_image(filepath)
        prediction = model.predict(image)
        class_idx = np.argmax(prediction)
        label = classes[class_idx]

        return render_template('index.html', label=label)

if __name__ == '__main__':
    app.run(debug=True)