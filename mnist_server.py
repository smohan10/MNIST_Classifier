"""
Server Script: REST API using Python's Flask 
"""

import logging
import time
import argparse
import json
import os
import numpy as np 
from flask import Flask, jsonify, request
from keras.models import model_from_json
from PIL import Image

# global variables
app = Flask(__name__)
model = None
logger = None


def load_model(config):
	"""
	Function to load the predefined model
	Arguments:
		config -- user argument config class of model information
	Returns:
		None
	"""

	global model
	model = model_from_json(open(config['model_architecture']).read())
	model.load_weights(config['model_weight'])
	
	

def create_logger_instance():  
    """
    Function to create an instance of logging at file and console level.
    Arguments:
    	None
    Returns:
    	None    
    """

    global logger

    # Create a logging folder
    if not os.path.isdir("logs"):
    	os.mkdir("logs")

    # create logger with 'Training_Automation'
    logger = logging.getLogger('MNIST Predict')
    logger.setLevel(logging.DEBUG)
    
    logging_name = 'logs/mnist_server_logs_' + str(int(time.time())) +  '.log'
                
    # create file handler which logs debug messages 
    fh = logging.FileHandler(logging_name)
    fh.setLevel(logging.DEBUG)
    
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)   



@app.route('/predict', methods=["POST"])
def predict_image():
	"""
	Function to predict the labels for an incoming image request
	Arguments: 
		None
	Returns:
		JSON formatted result
	"""


	# Open the image from the request, reshape and normalize the image
	image = request.files['file']
	logger.debug("This is the image request: %r" % image)
	image = Image.open(image)
	image = np.asarray(image.resize((28,28)))
	image = image.reshape(1,28,28,1)
	image = image/255

	# Predict the output using the Keras model's predict method
	pred = model.predict(image)
	predicted_output_label = np.argmax(pred)

	# Compute the result and prepare the output to send to the client	
	prediction = {'predicted_output_label':int(predicted_output_label), 'probability':float(max(pred[0]))}

	logger.debug("The prediction for above request: %r\n\n" % prediction)

	return jsonify(prediction)


if __name__ == "__main__":

	"""
	Main function that receives config file from the user, creates logging instance, 
	loads the model and starts the server
	"""

	parser = argparse.ArgumentParser(description="MNIST Server")
	parser.add_argument("--config_file", default="model_config.cfg")		
	args = parser.parse_args()

	# Create a logging instance
	create_logger_instance()

	# Open the config file
	with open(args.config_file, "r") as cfg:
		config = json.load(cfg)
	logger.debug("Config file contents are: %r" % config)

	# Load the trained model
	load_model(config)

	# Start the server
	app.run()