import io
import numpy as np
from PIL import Image
from scipy.stats import mode
from flask import Flask, request, jsonify

from model import Models

CRITERIA = ["lithology", "SW_fragments", "morphology"]
ARCHITECTURES = ["Vgg", "Xception"]

class_dict = {"lithology":    {"class_number": 3, "class_names": ["Slab", "Sulfurs", "Volcanoclastic"]},
             "SW_fragments": {"class_number": 3, "class_names": ["0-10%", "10-50%","50-100%"]},
             "morphology":   {"class_number": 4, "class_names": ["Fractured", "Marbled", "ScreeRubbles","Sedimented"]}
             }

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
    global MODELS
    print("fue llamado")
    print(FILE_NAMES)
    print(MODELS)
    images = request.files.getlist("image")
    results = []
    for i, file in enumerate(images):
        img = np.array(Image.open(file.stream).convert("RGB"))[np.newaxis, :,:,:]
        results_image = []
        # Preprocessing the image
        counter = 0
        for architecture, model_instance in MODELS.items():
            predictions = model_instance.predict(img, architecture)
            # Storing the predictions in a matrix per image in the way of [number of predictions, class]
            if results_image == []:
                results_image = np.zeros((len(ARCHITECTURES) * len(predictions), predictions[0].shape[1]))
            for prediction in predictions:
                results_image[counter, :] = prediction[0]
                counter += 1
        # Converting each prediction to a class in binary format
        results_image_c = np.argmax(results_image, axis=1)
        # Applying majority voting to the predictions
        majority_voting = mode(results_image_c, axis=0)[0]
        # Computing the predictive entropy of the predictions likelihood
        predicitve_entropy = -1 * np.mean(np.mean(results_image, axis=0) * np.log(np.mean(results_image, axis=0) + 1e-10))
        print(predicitve_entropy)
        # Storing the results in a dictionary
        results.append({
            "filename": FILE_NAMES[i],
            "predictions": {
                CRITERIA[0]: {
                    "class_number": class_dict[CRITERIA[0]]["class_number"],
                    "predicted_index": int(majority_voting[0]),
                    "uncertainty": float(predicitve_entropy),
                    "predicted_class": class_dict[CRITERIA[0]]["class_names"][int(majority_voting[0])]
                }
            }
        })
    print("Predictions finished successfully!!!")
    FILE_NAMES = []
    return jsonify({'results': results})


if __name__ == '__main__':
    # Initializing the previously trained segmentation models
    
    global MODELS
    MODELS = {}
    for architecture in ARCHITECTURES:
        print("Loading model for: ", CRITERIA[0], " and architecture: ", architecture)
        models = Models(criteria=CRITERIA[0], architecture=architecture)
        MODELS[architecture] = models
    print("Models loaded successfully")
    print("Starting Flask server...")
    # Start the Flask server
    app.run(debug = False, host = '0.0.0.0', port = 5500)



