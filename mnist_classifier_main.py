import argparse
from mnist_classifier_utils import *

if __name__ == "__main__":

	"""
	Main function that trains the MNIST data using the CNN architecture
	"""

	parser = argparse.ArgumentParser(description="This is MNIST CNN Model")
	parser.add_argument("--epochs", default="10")
	parser.add_argument("--batch_size", default="128")
	parser.add_argument("--validation", default="0.2")

	args = parser.parse_args()

	(X_train_norm, y_train_one_hot, X_test_norm, y_test_one_hot, input_shape, output_labels) = prepare_mnist_data()

	# Define the variables for training
	validation_split_percent = float(args.validation)
	num_epochs = int(args.epochs)
	batch_size = int(args.batch_size)

	print("Validation split percentage is %f" % validation_split_percent)
	print("Number of epochs is %d" % num_epochs)
	print("Batch size is %d" % batch_size)

	# Create a CNN model for the MNIST Data
	model = conv_model(input_shape, output_labels)

	# Print the model summary, describing the layers and dimentions of each layer
	model.summary()

	# Compile the model with Adam optimizer using cross entropy loss function
	model.compile(loss = keras.losses.categorical_crossentropy, 
	              optimizer= "Adam", 
	              metrics=['accuracy'])

	# Fit the training images and labels keeping 20% validation data aside, run against 10 epochs and batch size of 128 
	model.fit(X_train_norm, y_train_one_hot, validation_split=validation_split_percent, epochs=num_epochs, batch_size=batch_size)

	# Evaluate the model against test images and labels
	preds = model.evaluate(x=X_test_norm, y=y_test_one_hot)
	
	# Print the results
	print("Training set loss after %d epochs is %f" % (num_epochs, model.history.history['loss'][-1]))
	print("Training set accuracy after %d epochs is %f" % (num_epochs, model.history.history['acc'][-1]))
	print("Test set loss is %f" % preds[0])
	print("Test set accuracy is %f" % preds[-1])


	# Save the model
	save_mnist_model(model, num_epochs, batch_size, validation_split_percent)

	print("End of the program")
	


