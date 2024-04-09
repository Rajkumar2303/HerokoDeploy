import numpy as np
import requests
from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
from keras.preprocessing import image as keras_image
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator



app = Flask(__name__)

# Load the LightGBM model
model = tf.keras.models.load_model('dsModel.h5')

# Define the label encoder or preprocessing steps if needed
label = {0: 'BrainTumour', 1: 'Normal',2: 'Stroke',3: 'Alzheimer'}




@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get the JSON data from the request
            
            data = request.get_json()
            # Extract the URL from the JSON data
            image_url = data.get('url')
            
            if image_url:
                # Fetch the image from the URL
                response = requests.get(image_url)
                
                if response.status_code == 200:
                    print('Image Received')
                    # Read the image from the response content
                    img = Image.open(BytesIO(response.content))
                    # Preprocess the image
                    img_gray = img.resize((224, 224))
                    img=img_gray.convert('RGB')
                    print('Image converted to RGB')

                    
                    
                    img_array = keras_image.img_to_array(img)
                    print(img_array.shape)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = img_array / 255.0 
                    tes_df = pd.DataFrame({'image_data': [img_array]})
                    tes_datagen = ImageDataGenerator()
                    img_data_array = np.array([tes_df['image_data'][0]])
                    img_data_array = np.squeeze(img_data_array, axis=1)
                    print(img_data_array.shape)
                    tes_gen = tes_datagen.flow(img_data_array, shuffle=False, batch_size=1)
                    print('tes_gen')
                    y_pred = model.predict(tes_gen)

                    print('model Predicted')
                    ind = int(np.argmax(y_pred, axis=1))
                        
                    # Determine the predicted class
                    predicted_class = label[ind]

                    # Return the prediction result as JSON
                    return jsonify({'predicted_Disease': predicted_class})
                else:
                    return jsonify({'error': 'Failed to fetch image from the URL'})
            else:
                return jsonify({'error': 'No image URL provided'})
        except Exception as e:
            return jsonify({'error': str(e)})
    else:
        return jsonify({'error': 'Invalid request method'})

if __name__ == '__main__':
    app.run(debug=True)
