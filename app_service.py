import io
import sys
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
    global MODELS_
    #print("fue llamado")
    #print(FILE_NAMES)
    #print(MODELS_)
    images = request.files.getlist("image")
    results = []
    for i, file in enumerate(images):
        img = np.array(Image.open(file.stream).convert("RGB"))[np.newaxis, :,:,:]
        # Preprocessing the image
        for CRITERIA_, MODELS  in MODELS_.items():
            counter = 0
            results_image = []
            print("Processing image: ", FILE_NAMES[i], " for criteria: ", CRITERIA_)
            for architecture, model_instance in MODELS.items():
                predictions = model_instance.predict(img, architecture)
                # Storing the predictions in a matrix per image in the way of [number of predictions, class]
                if results_image == []:
                    results_image = np.zeros((len(ARCHITECTURES) * len(predictions), predictions[0].shape[1]))
                for prediction in predictions:
                    results_image[counter, :] = prediction[0]
                    counter += 1
            #print(results_image)
            # Converting each prediction to a class in binary format
            if CRITERIA_ != CRITERIA[2]:
                results_image_c = np.argmax(results_image, axis=1)
                #Applying majority voting to the predictions
                majority_voting = mode(results_image_c, axis=0)[0]
                #Computing the predictive entropy of the predictions likelihood
                predicitve_entropy = -1 * np.mean(np.mean(results_image, axis=0) * np.log(np.mean(results_image, axis=0) + 1e-10))
                #print(predicitve_entropy)
                #Storing the results in a dictionary
                if len(results) == 0:
                    results.append({
                        "filename": FILE_NAMES[i],
                        "predictions": {
                            CRITERIA_: {
                                "class_number": class_dict[CRITERIA_]["class_number"],
                                "predicted_index": int(majority_voting[0]),
                                "uncertainty": float(predicitve_entropy),
                                "predicted_class": class_dict[CRITERIA_]["class_names"][int(majority_voting[0])]
                            }
                        }
                    })
                elif results[-1]["filename"] == FILE_NAMES[i]:
                    results[-1]["predictions"][CRITERIA_] = {
                        "class_number": class_dict[CRITERIA_]["class_number"],
                        "predicted_index": int(majority_voting[0]),
                        "uncertainty": float(predicitve_entropy),
                        "predicted_class": class_dict[CRITERIA_]["class_names"][int(majority_voting[0])]
                    }
                else:
                    results.append({
                        "filename": FILE_NAMES[i],
                        "predictions": {
                            CRITERIA_: {
                                "class_number": class_dict[CRITERIA_]["class_number"],
                                "predicted_index": int(majority_voting[0]),
                                "uncertainty": float(predicitve_entropy),
                                "predicted_class": class_dict[CRITERIA_]["class_names"][int(majority_voting[0])]
                            }
                        }
                    })
            else:
                #Storing the results in a dictionary
                results_image_c = ((results_image > 0.5) * 1.0)
                majority_voting_ = np.sum(results_image_c, axis=0)
                majority_voting = [i for i, x in enumerate(majority_voting_) if x > 4]
                for j in range(len(majority_voting)):
                    if j == 0:
                        indexs = str(majority_voting[j])
                        class_name = class_dict[CRITERIA_]["class_names"][int(majority_voting[j])]
                    else:
                        indexs += ", " + str(majority_voting[j])
                        class_name += ", " + class_dict[CRITERIA_]["class_names"][int(majority_voting[j])]
                results[-1]["predictions"][CRITERIA_] = {
                    "class_number": class_dict[CRITERIA_]["class_number"],
                    "predicted_index": indexs,
                    "uncertainty": 0.0,
                    "predicted_class": class_name,
                }
    print("Predictions finished successfully!!!")
    FILE_NAMES = []
    return jsonify({'results': results})


if __name__ == '__main__':
    # Initializing the previously trained segmentation models
    
    global MODELS
    MODELS_ = {}
    # Load the models for lithology and the different architectures
    MODELS_LITHOLOGY = {}
    for architecture in ARCHITECTURES:
        print("Loading model for: ", CRITERIA[0], " and architecture: ", architecture)
        MODELS_LITHOLOGY[architecture] = Models(criteria=CRITERIA[0], architecture=architecture)
    
    MODELS_SW = {}
    for architecture in ARCHITECTURES:
        print("Loading model for: ", CRITERIA[1], " and architecture: ", architecture)
        MODELS_SW[architecture] = Models(criteria=CRITERIA[1], architecture=architecture)

    MODELS_MORPHOLOGY = {}
    for architecture in ARCHITECTURES:
        print("Loading model for: ", CRITERIA[2], " and architecture: ", architecture)
        MODELS_MORPHOLOGY[architecture] = Models(criteria=CRITERIA[2], architecture=architecture)

    # Storing the models in a dictionary
    MODELS_ = {
        CRITERIA[0]: MODELS_LITHOLOGY,
        CRITERIA[1]: MODELS_SW,
        CRITERIA[2]: MODELS_MORPHOLOGY
    }
        
    print("Models loaded successfully")
    print("Starting Flask server...")
    # Start the Flask server
    app.run(debug = False, host = '0.0.0.0', port = 5500)



