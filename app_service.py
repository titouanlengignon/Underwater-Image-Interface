import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/filenames', methods = ['POST'])
def get_filenames():
    global FILE_NAMES
    FILE_NAMES = []
    file_names_ = []
    file_names = request.get_json()
    for filename in file_names:
       FILE_NAMES.append(filename[1])
    print(FILE_NAMES)
    return file_names_

@app.route('/predict', methods=['POST'])
def get_images():
    global FILE_NAMES
    print("fue llamado")
    print(FILE_NAMES)
    images = request.files.getlist("image")
    results = []
    for i, file in enumerate(images):
        img = np.array(Image.open(file.stream).convert("RGB"))[np.newaxis, :,:,:]

    print("Predictions finished successfully!!!")
    FILE_NAMES = []
    return jsonify({'results': results})


if __name__ == '__main__':
    # Initializing the previously trained segmentation models 
    app.run(debug = False, host = '0.0.0.0', port = 5500)



