""" Backend server """

##### Libraries 

import base64
import io
import joblib
import os
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import sys
from flask import Flask, render_template, request, jsonify
from PIL import Image

# =============================================================================
# Directory Paths
# =============================================================================

CURRENT_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.dirname(CURRENT_SCRIPT_PATH)
TRAINED_NETWORK_PATH = os.path.join(ROOT_PATH, 'doodle', 'trained_network')
ENCODER_PATH = os.path.join(ROOT_PATH, 'data', 'joblib')
sys.path.append(ROOT_PATH)

# =============================================================================
# Backend
# =============================================================================
from doodle.model.doodle_network import DoodleANN

# Start app
app = Flask(__name__)

# Load the model
model = DoodleANN()
model.load_state_dict(torch.load(os.path.join(TRAINED_NETWORK_PATH, 'doodle_ann_weights_1000_2layers.pth')))
model.eval()

# Load encoder
encoder = joblib.load(os.path.join(ENCODER_PATH, 'doodle_category_encoder.joblib'))

@app.route('/')
def index():
    # Render the main page with the drawing interface
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    # Decode the image data
    image_data = request.json['image_data']
    image_data = image_data.split(",")[1]  # Remove the "data:image/png;base64," part
    image_data = base64.b64decode(image_data)

    # Convert binary data to PIL Image
    image = Image.open(io.BytesIO(image_data))

    # Preprocess the image
    transform = transforms.Compose([
        transforms.Grayscale(),              # Convert the image to grayscale
        transforms.Resize((28, 28)),         # Resize the image to 28x28 pixels
        transforms.ToTensor(),               # Convert the image to a torch tensor
        transforms.Lambda(lambda x: 1 - x)   # Invert colors: make black white and white black
    ])

    image_tensor = transform(image).unsqueeze(0)
    
    # Predict with the model
    model.eval()  # Ensure the model is in eval mode
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs.data, 1)
    
    # softmax_outputs = F.softmax(outputs, dim=1)
    # print(f"outputs: {outputs}")
    # print(f"Softmaxed outputs: {softmax_outputs}")
    # print(f"Sum of softmax outputs = {softmax_outputs.sum()}")
    # print(f"predicted: {predicted}")
    # Convert class index to label
    predicted_labels = predicted.numpy()  
    predicted_class = encoder.inverse_transform(predicted_labels)

    return jsonify({'class': predicted_class[0]})

if __name__ == '__main__':
    app.run(debug=True)
