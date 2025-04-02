import os
import sys
import argparse
import numpy as np
import pandas as pd
from flask import jsonify
import tensorflow as tf
from Networks import Networks     # Load the model architecture and weights

parser = argparse.ArgumentParser(description='Underwater Image Classification')

parser.add_argument('--train_task', type=str, default='Image_Classification', help='Task to be trained')
parser.add_argument('--learning_model', type=str, default='CNN', help='Learning model to be trained')
parser.add_argument('--backbone_name', type=str, default='None', help='Backbone name')
parser.add_argument('--pretrained_backbone', type=bool, default=False, help='Use pretrained backbone')
parser.add_argument('--new_size_rows', type=int, default=1024, help='New size rows')
parser.add_argument('--new_size_cols', type=int, default=1024, help='New size columns')
parser.add_argument('--image_channels', type=int, default=3, help='Number of image channels')
parser.add_argument('--labels_type', type=str, default='onehot_labels', help='Labels type')
args = parser.parse_args()

clss_dict = {"lithology":    {"class_number": 3, "class_names": ["Slab", "Sulfurs", "Volcanoclastic"]},
             "SW_fragments": {"class_number": 3, "class_names": ["0-10%", "10-50%","50-100%"]},
             "morphology":   {"class_number": 4, "class_names": ["Fractured", "Marbled", "ScreeRubbles","Sedimented"]}
             }

class Models():
    def __init__(self, criteria = "lithology", architecture = "Vgg"):
        self.args = args
        self.SESSIONS = []
        ROOT_DIR = "./trained_models/"
        self.criteria = criteria
        self.architectures = architecture

        self.data = tf.placeholder(tf.float32, shape=[None, self.args.new_size_rows, self.args.new_size_cols, 3], name ='input_data')
        
       
        args.class_number = clss_dict[criteria]["class_number"]     
        model_path = os.path.join(ROOT_DIR, criteria, architecture)
        cnns_names = os.listdir(model_path)
        for c, cnn_name in enumerate(cnns_names):
            cnn_path = os.path.join(model_path, cnn_name)
            print(cnn_path)
            if os.path.exists(model_path):
                args.backbone_name = architecture
                model = Networks(args)
                if c != 0:
                    print("second model")
                    classifier_output = model.learningmodel.build_Model(input_data=self.data, reuse = True, name="CNN")
                else:
                    print("first model")
                    classifier_output = model.learningmodel.build_Model(input_data=self.data, reuse = False, name="CNN")
            self.prediction_c = classifier_output[-1]
            ## Load the model weights
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            print('[INFO] Loading model from: {}'.format(cnn_path))
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(cnn_path)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                saver.restore(sess, os.path.join(cnn_path, ckpt_name))
                print('[INFO] Model loaded successfully')
            else:
                print('[ERROR] Model not found')
                continue
            self.SESSIONS.append(sess)
        tf.compat.v1.reset_default_graph()

    def predict(self, image, architecture):
        """
        Predict the class of the input image using the loaded models.
        """
        print("Predicting using architecture: ", architecture)
        print(np.shape(image))
        results = []
        print("Number of sessions: ", len(self.SESSIONS))
        image = self.preprocess_input(image)
        for i, sess in enumerate(self.SESSIONS):
            print("Predicting using session: ", i)
            predictions = sess.run(self.prediction_c, feed_dict={self.data: image})
            results.append(predictions) 
        return results
    
    def preprocess_input(self, image):
        """
        Preprocess the input image for prediction.
        """
        # Perform any necessary preprocessing steps here
        # For example, resizing, normalization, etc.
        resized_image = np.zeros((1, self.args.new_size_rows, self.args.new_size_cols, self.args.image_channels))
        resized_image[0,:,:,:] = np.resize(image, (self.args.new_size_rows, self.args.new_size_cols, self.args.image_channels))
        normalized_image = resized_image / 255.0
        return normalized_image