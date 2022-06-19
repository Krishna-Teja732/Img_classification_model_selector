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
image, labels = iterator.next(self, scale_bottom=0, flip_images = False, light_change = False, crop_images = False, save_processed_img = False, save_img_path = "/", normalization=True)

# CustomDirectoryIterator.test_next() method returns
# a tuple containing the images of given batch 
# size and the correspoing labels of the images
# (the image labels are Label encoded). This method 
# is used for obtaining the test dataset
image, labels = iterator.test_next(scale_bottom=0)
```

## 