from keras.datasets import mnist
import logging
import time
import datetime as dt
import os
import sys
import numpy as np
from matplotlib.pyplot import imshow, show
import random
import math
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, concatenate
from keras.layers import Input, Activation, Dense, Dropout, Flatten, BatchNormalization


def display_data(x_train, y_train, x_test, y_test):
    """
    Function to display the image and the label.    
    Arguments:
        x_train -- Training data
        y_train -- Training label
        x_test  -- Test data
        y_test  -- Test label    
    Returns: 
        None
    """


    # Randomly select an image and label from train and test
    random_sample = random.randint(0,np.shape(x_train)[0]-1)
    print("Sample train image: ")
    imshow(x_train[random_sample])
    show()
    print("Label of the train sample image: ", y_train[random_sample])

    random_sample = random.randint(0,np.shape(x_test)[0]-1)
    print("\n\nSample test image: ")
    imshow(x_test[random_sample])
    show()
    print("Label of the test sample image: ", y_test[random_sample])


def prepare_mnist_data():
    """
    Function to prepare data for training
    Arguments:
        None
    Returns:
        Tuple object of the data prepared for training
    """

    
    # Load the training and test data from MNIST
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print("This is the MNIST Data:")
    print("The shape of X train: {}".format(np.shape(x_train)))
    print("The shape of X test: {}".format(np.shape(x_test)))
    print("The shape of Y train: {}".format(np.shape(y_train)))
    print("The shape of Y test: {}".format(np.shape(y_test)))

    # Extract the shape of the training and test data
    m_train, nH, nW = np.shape(x_train)
    m_test, nH, nW = np.shape(x_test)
    input_shape = (nH, nW, 1)

    # Normalize and reshape the inputs
    X_train_norm = x_train/255
    X_train_norm = X_train_norm.reshape([m_train, nH, nW, 1])
    X_test_norm = x_test/255
    X_test_norm = X_test_norm.reshape([m_test, nH, nW, 1])

    # Create one hot encoded labels for train and test
    ny = max(y_train)+ 1     # num of classes
    y_train_one_hot = keras.utils.to_categorical(y_train, ny)
    y_test_one_hot = keras.utils.to_categorical(y_test, ny)


    # Print a sample train and test labels
    random_sample_train = random.randint(0,m_train-1)
    random_sample_test = random.randint(0,m_test-1)
    print("One hot encoding for training label {} is {}".format(y_train[random_sample_train], y_train_one_hot[random_sample_train,:]))
    print("One hot encoding for test     label {} is {}".format(y_test[random_sample_test]  , y_test_one_hot[random_sample_test,:]))

    return (X_train_norm, y_train_one_hot, X_test_norm, y_test_one_hot, input_shape, ny)



def conv_model(input_shape, ny):
    """
    Function: Modeling MNIST data using Convolution Neural Network Architecture

                            -> Conv -> Relu   -> Conv -> Relu
    Model: X -> Conv ->                                +        -> Concatenation -> Dense (1000) -> Dense (500) -> Output (10)
                            -> Conv -> Relu   -> Conv -> Relu
    
    Arguments:
        input_shape -- Shape of the input image [?, 28, 28, 1]
        ny -- Output classes (10)

    Returns:
        model -- Keras model based on the functions architecture
    """
    
    # Define the input
    X_input = Input(shape=input_shape)
    
    # Conv1_1
    X = Conv2D(filters=32, kernel_size=(3,3), strides= (1,1), padding='SAME')(X_input)
    X_Conv1_1 = Activation('relu')(X)
    
    # Conv2_1
    X_Conv2_1 = Conv2D(filters=64, kernel_size=(3,3), strides= (1,1), padding='SAME')(X_Conv1_1)
    X_Conv2_1 = Activation('relu')(X_Conv2_1)
    X_Conv2_1 = MaxPooling2D(pool_size=(2,2))(X_Conv2_1)

    # Conv2_2
    X_Conv2_2 = Conv2D(filters=64, kernel_size=(3,3), strides= (1,1), padding='SAME')(X_Conv1_1)
    X_Conv2_2 = Activation('relu')(X_Conv2_2)
    X_Conv2_2 = MaxPooling2D(pool_size=(2,2))(X_Conv2_2)
    
    # Conv3_1
    X_Conv3_1 = Conv2D(filters=512, kernel_size=(3,3), strides= (1,1), padding='SAME')(X_Conv2_1)
    X_Conv3_1 = Activation('relu')(X_Conv3_1)
    X_Conv3_1 = MaxPooling2D(pool_size=(2,2))(X_Conv3_1)

    # Conv3_2
    X_Conv3_2 = Conv2D(filters=512, kernel_size=(3,3), strides= (1,1), padding='SAME')(X_Conv2_2)
    X_Conv3_2 = Activation('relu')(X_Conv3_2)
    X_Conv3_2 = MaxPooling2D(pool_size=(2,2))(X_Conv3_2)

    # Merge 3rd convoluation layers
    X = concatenate([X_Conv3_1, X_Conv3_2])

    # Flatten the input
    X = Flatten()(X)
    
    # Fully connected layer 1
    X = Dense(1000, activation='relu')(X)
    
    # Fully connected layer 2
    X = Dense(500, activation='relu')(X)
    
    # Output Layer (ny dimentions)
    X = Dense(ny, activation='softmax')(X)
    
    # Create an instance of the model with the inputs and the outputs
    model = Model(inputs=X_input, outputs=X)
    
    return model



def save_mnist_model(model, num_epochs, batch_size, validation_split_percent):

    if not os.path.isdir("models"):
        os.mkdir("models")

    model_params_string = str(num_epochs) + "_" + str(batch_size) + "_" \
                                + str(int(validation_split_percent*100)) + "_" + str(int(time.time()))
    
    model_architecture_name = "models/model_arch_" + model_params_string + ".json"

    model_weights_name = "models/model_weights_" + model_params_string + ".h5"

    model_in_json = model.to_json()
    with open(model_architecture_name, "w") as mj:
        mj.write(model_in_json)

    model.save_weights(model_weights_name)
    del model
    print("Saved the models to disk")
