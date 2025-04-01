import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify

from model import Models

app = Flask(__name__)

@app.route('/filenames', methods = ['POST'])
def get_filenames():
    global FILE_NAMES
    FILE_NAMES = []
    file_names = request.get_json()
    for filename in file_names:
       FILE_NAMES.append(filename[1])
    return jsonify({'results': 0})

@app.route('/predict', methods=['POST'])
def get_images():
    global FILE_NAMES
    print("fue llamado")
    print(FILE_NAMES)
    images = request.files.getlist("image")
    results = 0
    for i, file in enumerate(images):
        img = np.array(Image.open(file.stream).convert("RGB"))[np.newaxis, :,:,:]
        print(np.shape(img))

    print("Predictions finished successfully!!!")
    FILE_NAMES = []
    return jsonify({'results': results})


if __name__ == '__main__':
    # Initializing the previously trained segmentation models
    criteria = ["lithology", "SW_fragments", "morphology"]
    architectures = ["Vgg", "Xception"]
    MODELS = {}
    for architecture in architectures:
        print("Loading model for: ", criteria[0], " and architecture: ", architecture)
        models = Models(criteria=criteria[0], architecture=architecture)
        MODELS[architecture] = models
    print("Models loaded successfully")
    print("Starting Flask server...")
    # Start the Flask server
    app.run(debug = False, host = '0.0.0.0', port = 5500)



