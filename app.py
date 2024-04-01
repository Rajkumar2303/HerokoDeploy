# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 02:04:55 2024

@author: rajku
"""



from __future__ import division, print_function
# coding=utf-8

import os


import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer


from PIL import Image

import tensorflow as tf


# Define a flask app
app = Flask(__name__)

# Load your trained model

MODEL_PATH ='newmodel.h5'

 

def load_model():

    try:

        model = tf.keras.models.load_model(MODEL_PATH)

        return model

    except Exception as e:

        print("Error loading the model:", str(e))

        return None

model = load_model()
# Model saved with Keras model.save()
'''MODEL_PATH ='model_resnet50.h5'

# Load your trained model

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print("Error loading the model:", str(e))
#model = load_model(MODEL_PATH)
'''



def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    
    x = np.expand_dims(x, axis=0)
   

   

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="BrainYumour"
    elif preds==1:
        preds="Normal"
    elif preds==2:
        preds="Stroke"
    else:
        preds="Azhzeimer"
    
    
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)