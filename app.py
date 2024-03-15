import numpy as np
import requests
from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
import joblib
import cv2

from keras.preprocessing import image as keras_image

app = Flask(__name__)

# Load the LightGBM model
model = joblib.load('BrainAnamolies.pkl')

# Define the label encoder or preprocessing steps if needed
label = {0: 'Brain Tumor', 1: 'NORMAL', 2: 'Stroke', 3: 'alzhzeimer'}




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
                    # Read the image from the response content
                    img = Image.open(BytesIO(response.content))
                    # Preprocess the image
                    img = img.resize((224, 224))

                    image = img.reshape(1, 224, 224, 3)

                    
                    # Normalize the image
                   
                   
                    # Make prediction using the model
                    prediction = model.predict(img)

                    # Convert NumPy array to Python list
                    prediction_list = prediction.tolist()
                    # Determine the predicted class
                    predicted_class = label[np.argmax(prediction_list)]

                    # Return the prediction result as JSON
                    return jsonify({
                        'prediction': prediction_list,
                        'predicted_class': predicted_class
                    })
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
