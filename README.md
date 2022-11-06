# Image Classification Model Selector

This project aims to provide a simpler method to train and select neural network models for image classification.

## Features

* Facilitates in training and selecting models for image classification
* Facilitates in balancing multilabel image dataset
* Facilitates in image preprocessing(preprocessing such as normalization, image centering, cropping, image augmentation and light shifting)

## Installing required packages

The project requires python 3.10 and pip. Use `pipenv install` to install all the required packages

## Modules

* CustomDirectoryIterator - Used for loading, balancing, and preprocessing image dataset
* dataset_utils - used to filter the dataset(remove files that are not images and remove images that can cause errors while reading it)
* CustomModel - `tensorflow.keras.Model` that is used to save tensorflow hub models locally
* CustomTuner - class that extends `keras_tuner.RandomSearch` which is used for hyperparameter tuning(tune learning rate, optimizer, and epochs to train model)
* frame_extractor - used to convert videos to image frames
* ModelSelector - used for loading, training, and testing image classification models
* test.py - provides an example on how to use model selector

## CustomDirectoryIterator

### Features

* image dataset preprocessing - normalization, image centering, image cropping, light shifting, and image rotations
* multilabel image dataset balancing - if a class of image has less number of instances, then random oversampling is used to populate the class of image
* load images batchwise - provides methods to load image batchwise which facilitates efficient memory usage

### Examples

``` python

# This is a static method used to balance image dataset
CustomDirectoryIterator.balance_images(path = 'dataset_path')


# To create a CustomDirectoryIterator, pass the
# image dataset path, the image size which is a
# tuple containing (height, width, number_of_channels)
# and the ratio to split the dataset for training and testing 
iterator = CustomDirectoryIterator(path = 'image_dataset_path', image_size = (300,300,3), batch_size = 32, training_size = 0.8)


# CustomDirectoryIterator.next() method returns
# a tuple containing the images of given batch 
# size and the correspoing labels of the images
# (the image labels are Label encoded). By 
# setting any argument to True the 
# corresponding image preprocessing is 
# performed on the images. The save_img_path and 
# save_preprocessed_img arguments are used to save 
# the preprocessed images
image, labels = iterator.next(scale_bottom=0, flip_images = False, light_change = False, crop_images = False, save_processed_img = False, save_img_path = "/", normalization=True)

# CustomDirectoryIterator.test_next() method returns
# a tuple containing the images of given batch 
# size and the correspoing labels of the images
# (the image labels are Label encoded). This method 
# is used for obtaining the test dataset
image, labels = iterator.test_next(scale_bottom=0)
```

## dataset_utils

### Example
``` python
from dataset_utils import ImageDataSetFilter

# This class removes all the files that cannot be loaded using pillow package
filter = ImageDataSetFilter(path = 'path_to+image_dataset')
```

## CustomModel

### Classes

* CustomModel - This class extends `tensorflow.keras.Model` which is used for saving Tensorflow hub pretrained models locally with the output layer
* BaseModel - This class extends `tensorflow.keras.Model` which is used for saving Tensorflow hub pretrained models locally without the output layer

### Methods
* max_ind(list) - returns the index containing the macimum index
* predict_class(list) - given the probabilities of various classes, it returns the class with maximum probability

### Example

``` python
from CustomModel import CustomModel, BaseModel
import tensorflow as tf

handle = 'https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5'

# downloads the Mobilenet Tensorflow hub model and 
# saves it locally at the given path. The saved 
# model does not have a output layer
BaseModel(handle, (224,224,3), './mobi_net_base')

# Use the base model and add an output layer based 
# on the number of classes in the image dataset. The 
# activation argument sets the activation function of the 
# output layer
CustomModel(base_model_path = './mobi_net_base', 
            save_model_path = './mobi_net',
            number_pf_classes = 3, 
            input_shape = (224,224,3),
            activation = tf.nn.softmax
            )

# Load a CustomModel that was saved locally 
model = tensorflow.keras.models.load('./mobi_net')

model.summary()
```

## ModelSelector

Used to train and test multiple keras models on the same image dataset.

### Features

* train and test multiple keras models on an image dataset
* Hyperparameter tuning
* Hard Voting using the multiple models

### Prerequsites

Before running model selector, the BaseModels need to be saved locally. The BaseModels can be downloaded and saved locally using download_basemodels() method in ModelSelector class

### Example

``` python
from sklearn.metrics import accuracy_score
from pprint import pprint
import ModelSelector as ms
from os import path

if __name__=='__main__':

    # Using only mobilenet model
    models = dict()
    key, val = ms.base_models.popitem()
    models[key] = val

    # creating an object of ModelSelector class. 
    # The image dataset path, models, model image 
    # input shape, and path to save model locally 
    # is provided as arguments
    selector = ms.ModelSelector(path.join('.','data','const data test'), 
                                models, ms.input_shape, 
                                save_model_path= path.join('.', 'const_models' , 'saved_models'))

    # loads all the models and creates 
    # CustomDirectoryIterator for each model
    selector.load_models(load_from_local = False)

    # Train all the models with hyperparameter tunings
    selector.train_models(tune_hyperparameters=True, max_trials= 1)

    selector.test_models()

    # Printing metrics of each model. The accuracy 
    # score, precision, recall, and confusin matrix 
    # are the metrics used to test the models
    for key in selector.summary:
        print("Model: ", key)
        pprint(selector.summary[key])
    
    # hard voting
    y_true, y_pred = selector.predict(path.join('.','data','const data test'), path.join('.','data','saved_images'))

    # printing predicted values for hard voting
    for val,pred in zip(y_true, y_pred):
        print(val, pred)
```