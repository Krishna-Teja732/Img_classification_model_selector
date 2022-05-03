import os
import shutil
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from keras.preprocessing.image import DirectoryIterator, ImageDataGenerator

class CustomDirectoryIterator:
    def __init__(self, path, img_size, batch_size = 32, training_size = 0.8) -> None:
        self.PATH = path
        self.IMG_SIZE = img_size
        self.directory_iterator = None
        self.BATCH_SIZE = batch_size
        self.training_size = training_size
        self.classes = []
        self.load_img_from_dir(batch_size=batch_size)
        self.samples = self.directory_iterator.samples
        self.reset_iterations()
        
    def reset_iterations(self):
        '''
            resets the training and testing iterations
        '''
        self.train_iterations = int(self.training_size * (self.samples // self.BATCH_SIZE))
        self.test_iterations = int((1-self.training_size) * (self.samples // self.BATCH_SIZE))


    def load_img_from_dir(self, batch_size = 32, image_data_gen = ImageDataGenerator(samplewise_center=True)):
        path = Path(self.PATH)
        self.directory_iterator = DirectoryIterator(path,image_data_generator=image_data_gen, target_size=self.IMG_SIZE,batch_size=batch_size, dtype= int)
        d_class = self.directory_iterator.class_indices
        self.classes = ['' for _ in range(len(d_class))]
        for i in d_class:
            self.classes[d_class[i]] = i
        print('==============================================')
        print("CLASS : ", self.classes)
        print('==============================================')

    def label_encoder(self, label) -> int:
        return self.classes.index(self.get_label(label))

    def get_label(self, ls):
        for i in range(len(ls)):
            if ls[i] == 1:
                return self.classes[i]
        return ''

    def save_images(self, ls, folder_path=".\\output", clear=False):
        '''
            saves the images list to the given folder path
            clear - if True clears the folder each time called and saves the image
            else adds the images to the folder
        '''
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        else:
            if clear:
                shutil.rmtree(folder_path)
                os.makedirs(folder_path) #clearing the elements in the folder
        img_file_number = 1
        if not clear:
            #change the number according to the files in the List
            img_file_number = len(os.listdir(folder_path)) + 1
    
        for image in ls:
            tf.keras.utils.save_img(folder_path+"\\"+str(img_file_number)+".jpg", image)
            img_file_number += 1

    def get_flipped_images(self, images:list) -> list:
        flipped_images = []
        for image in images:
            flipped_images.append(tf.image.flip_left_right(image).numpy())
            flipped_images.append(np.array(image))
        return flipped_images

    def get_light_shifted_images(self, images:list) -> list:
        light_shifted_images = []
        light_range = (0.001,2.0)
        for image in images:
            light_shifted_images.append(tf.keras.preprocessing.image.random_brightness(image, light_range))
            light_shifted_images.append(image)
        return light_shifted_images

    def get_cropped_images(self, image, CROP_SIZE = 299) -> list:
        new_image = tf.image.resize(image, (CROP_SIZE+32, CROP_SIZE+32))
        CROP_SUB = 32
        #creating 5 crops
        image_list = []
        img = [i[:CROP_SIZE] for i in new_image[:CROP_SIZE]]
        image_list.append(img)
        img = [i[:CROP_SIZE] for i in new_image[CROP_SUB:]]
        image_list.append(img)
        
        img = [i[CROP_SUB:] for i in new_image[CROP_SUB:]]
        image_list.append(img)
        
        img = [i[CROP_SUB:] for i in new_image[:CROP_SIZE]]
        image_list.append(img)
        
        img = [i[CROP_SUB//2:CROP_SUB//2+CROP_SIZE] for i in new_image[CROP_SUB//2:CROP_SUB//2+CROP_SIZE]]
        image_list.append(img)
       
        return image_list


    #only normalization
    def next(self, scale_bottom=0, flip_images = False, light_change = False, crop_images = False, save_processed_img = False, save_img_path = "/", normalization=True) -> tuple:
        '''
            returns the next batch of images, labels tuple for training 
            runs until training iteration reaches 0
            when training iteration is 0 return "None, None"

            flip_images = on true does the horizontal flip on each image
            light_change = on true changes the brightness of the image randomly within the range (0.001, 0.2)
            crop_images = crops the images to 5 crops, best not used if not going for aggressive augmentation
        '''
        if self.train_iterations <= 0:
            self.reset_iterations()
            return None, None #end of current iteration

        self.train_iterations -= 1
        images , labels = self.directory_iterator.next()
        
        normalizer = tf.keras.layers.Rescaling(1./255)
        if scale_bottom != 0:
            normalizer = tf.keras.layers.Rescaling(1./127.5, offset=-1)
        if normalization == False:
          normalizer = tf.keras.layers.Rescaling(1)

        res_images = list(map(lambda x: normalizer(x), images))
        if crop_images: 
            cropped_images = []
            cropped_labels = []
            for i in range(len(res_images)):
                new_images = self.get_cropped_images(res_images[i], self.IMG_SIZE[0])
                cropped_images += new_images
                cropped_labels += [labels[i] for _ in range(len(new_images))]
            res_images = cropped_images
            labels = cropped_labels

        if flip_images:
            res_images = self.get_flipped_images(res_images)
        
        if light_change:
            res_images = self.get_light_shifted_images(res_images)

        if save_processed_img:
            self.save_images(res_images, save_img_path, True)
        
        return (
            np.array(res_images),
            np.array(list(map(self.label_encoder, labels)))
        )

    #using a different next function for the testset
    #as test iterations are calculated separately
    def test_next(self, scale_bottom = 0):
        '''
            returns a batch of image for testing
            batch size is given during construction
        '''
        if self.test_iterations <= 0:
            self.reset_iterations()
            return None, None
        self.test_iterations -= 1
        images , labels = self.directory_iterator.next()
        
        normalizer = tf.keras.layers.Rescaling(1./255)
        if scale_bottom != 0:
            normalizer = tf.keras.layers.Rescaling(1./127.5, offset=-1)

        images = list(map(lambda x: normalizer(x), images))

        return (
            np.array(images), np.array(list(map(self.label_encoder, labels)))
        )
