from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Folder to store uploaded images

# Load your pre-trained model
model = load_model('MobileNetV2_waste_classification.h5')  # Replace with your model filename

# Class names based on your dataset (e.g., Paper, Plastic, etc.)
class_names = ['Paper', 'Plastic', 'Metal', 'Glass', 'Cardboard', 'Vegetation', 'Food Organics', 'Textile Trash', 'Miscellaneous Trash']  # Adjust based on your labels

def predict_waste_type(img_path):
    # Load and preprocess the image
    img = load_img(img_path, target_size=(224, 224))  # Adjust target_size if needed
    img_array = img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match model input

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    return predicted_class

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Save the file to 'static/uploads'
            filepath = os.path.join('static', 'uploads', file.filename)
            file.save(filepath)
            
            # Make a prediction
            predicted_class = predict_waste_type(filepath)
            return render_template('result.html', prediction=predicted_class, image_path='uploads/' + file.filename)
    return render_template('index.html')

if __name__ == "__main__":
    # Create upload folder if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
