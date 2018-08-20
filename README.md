# MNIST_Classifier

The repository contains of software to perform the following tasks:

1. train MNIST data using Convolutional Neural Network Architecture
2. Host the trained model on a server using REST API

## Software & Libraries:

- Python
- Keras
- Flask API

## Machine:

Tested on Titan V GPU Machine. 

## Training

- The **mnist_classifier_main.py** script fetches the MNIST training and testing data.
- The input goes through 3 layers of convolution before passing through 2 fully connected layers and finally the output layer.

![Alt text](images/arch.png?raw=true "Title")

- The model is finally saved to disk for future references.

## How to run training?

The command to run the training script is

```
nohup python mnist_classifier_main.py --epochs 10 --batch_size 128 --validation 0.2 &
```

The default parameters are:
- Number of epochs: 10
- Batch size : 128
- Validation data split percentage: 20%

To run the training script with default parameters:

```
nohup python mnist_classifier_main.py &
```

Console output is written to **nohup.out**

The model architecture and weights are saved to 'models/' folder in the current directory of the script with parameters in the name and timestamp. 


## How to create a server which hosts the model?

A simple REST API is written that loads the model and hosts on its own server on port 5000.
The command to start the server is:

```
python mnist_server.py --config_file model_config.cfg
```

The script takes an argument that's a config file in a JSON format. The file has the paths of the model architecture and weights saved on disk during the training step. 

Server logs are created and located in the 'logs/' folder for reference.

Press ``` Ctrl + C ``` to stop the server. 

## How to feed an image to the server?

An example CURL command to call the predict method:

```
curl -F 'file=@/path_to_png_file/test.png' http://localhost:5000/predict
```

The client receives the predicted output label with the probability of prediction.



