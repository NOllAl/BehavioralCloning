
# coding: utf-8
"""
Created on Wed Jan 11 21:00:46 2017

@author: Alexander Noll
@description: Python script to train the model
"""
import os
import csv
import cv2
import numpy as np
import json
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dropout, Dense, Activation, Merge
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD
from keras import initializations
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

from preproc import preproc_image

# Hyper Params
BATCH_SIZE = 100
LEARNING_RATE = 0.00006
DROPOUT_RATE = 0.35
EPOCHS = 30
LOWER_ANGLE = -2
UPPER_ANGLE = 2
OFFSET = 0.25
FLIP_PROB = 0.5
REBALANCE_PROB = 0.5
TRANSLATION = 100
RANDOM_NOISE = 0.05

# Helper function to rotate images
def rotate_image(img, rotation_angle):
    """
    Takes an image and rotates it by a given angle.
    Returns rotated image
    """
    rows, cols = img.shape[0:2]
    M = cv2.getRotationMatrix2D((cols/2,rows/2), rotation_angle, 1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    dst = np.reshape(dst, (66, 208, 3))
    
    return(dst)

def rotate_all_images(images, LOWER_ANGLE, UPPER_ANGLE):
    """
    Rotates a list of images
    """
    return np.array([rotate_image(img, np.random.randint(LOWER_ANGLE, UPPER_ANGLE)) for img in images])

# Helper function to translate image
def translate_image(x, y):
    """
    Generate random translation of image and offset the steering angle
    """
    # Generate random translation
    tr_x = TRANSLATION * np.random.uniform() - TRANSLATION / 2
    y += tr_x / TRANSLATION * 2 * OFFSET
    M = np.float32([[1, 0, tr_x],[0, 1, 0]])
    x = cv2.warpAffine(x, M, (320, 160))
    return x, y

# Helper function to add random noise
def add_random_noise(x):
    """
    Add random noise to images
    """
    noise = np.random.normal(scale=RANDOM_NOISE, size=(66, 208, 3))
    x += noise
    return x



def process_line(line):
    """
    Processes a single line of the training data set and returns the three
    images and the steering angle
    
    :param line:
    """
    # Set input image paths
    center_img_path = (os.path.join('./data', line[0]))
    left_img_path = os.path.join('./data', line[1].strip())
    right_img_path = os.path.join('./data', line[2].strip())
    
    # Read Output
    y = np.reshape(np.array(np.float32(line[3].strip())), (1))  
    
    # Generate random integer between 0 and 2 to specify whether 
    # to take left (1), right(2) or center(0) image
    rand_int = np.random.randint(3)
    
    if rand_int == 1:
        # Left images are offset by +OFFSET
        x = cv2.imread(left_img_path)
        y += OFFSET
        x = np.reshape(x, (160, 320, 3))
    elif rand_int == 2:
        # Right images are offset by -OFFSET
        x = cv2.imread(right_img_path)
        y -= OFFSET
        x = np.reshape(x, (160, 320, 3))
    elif rand_int == 0:
        # Center images are translated at random and offset by a 
        # corresponding amount
        x = cv2.imread(center_img_path)
        x = np.reshape(x, (160, 320, 3))
        
    # Translate image    
    x, y = translate_image(x, y)
        
    # Flip images about vertical axis to generate more training images
    if np.random.rand() > FLIP_PROB:
        x = np.fliplr(x)
        y = -y  # Steering angle needs to be taken negative
        
    # Preprocess image
    x = preproc_image(x)
    
    # Add Gaussian random noise to image to enlarge training set
    #x = add_random_noise(x)
    
    return x, y

def generate_train(batch_size):
    """
    Generator used for training. Loops over training data indefinitely.
    
    :param path: directory where the training data are unzipped
    
    :returns: 
    """
    # Initialize return values
    output_x = []
    output_y = []
    # Infinite loop
    while 1: 
        with open('./data/driving_log.csv', newline='') as csvfile:
            count_ind = 0    # Initialize counting index
            csv_reader = csv.reader(csvfile, delimiter=',')
            next(csv_reader)  # Skip header
            
            # Iterate over training images
            for row in csv_reader:
                count_ind += 1
                if count_ind >= 40000:
                    # Leave csv reading loop after training set has been interated over
                    break
                if np.random.rand() > 0.999:
                    # Reshuffle training set by only taking every 1000th image at random
                    
                    # Process single line
                    x, y = process_line(row)
                    
                    # Rebalance training set by requiring output angle to be above 0.1
                    # or take a smaller angle with probability 1 - REBALANCE_PROB
                    if np.abs(y[0]) > 0.1 or (np.random.rand() > REBALANCE_PROB):
                        output_x.append(x)
                        output_y.append(y)
                
                # Only yield result if we are above the batch_size
                if len(output_x) >= batch_size:
                    # Put everyhing in right type and reshape
                    output_x =  np.array(output_x)
                    x_ret = np.reshape(output_x, (batch_size, 66, 208, 3))
                    y_ret = np.reshape(output_y, (batch_size))
                    
                    # Rotate images to enlarge training set
                    #x_ret = rotate_all_images(x_ret, LOWER_ANGLE, UPPER_ANGLE)
                    # Reinitialize output lists
                    output_x = []
                    output_y = []
                    yield (x_ret, y_ret)

def generate_val():
    """
    Generator used for validation and testing.
    """
    # Initialize return values
    output_x = []
    output_y = []
    skip = 30000
    # Infinite loop
    while 1: 
        with open('./data/driving_log.csv', newline='') as csvfile:
            count_ind = 0    # Initialize counting index
            csv_reader = csv.reader(csvfile, delimiter=',')
            next(csv_reader)  # Skip header
            
            # Skip training data
            for k in range(skip):
                next(csv_reader)

            # Iterate over images
            for row in csv_reader:
                count_ind += 1
                if count_ind >= 10000:
                    # Leave csv reading loop after cv set has been interated over
                    break
                # Process single line
                x, y = process_line(row)
                  
                
                # Only yield result if we are above the batch_size
                if len(output_x) >= batch_size:
                    # Put everyhing in right type and reshape
                    output_x =  np.array(output_x)
                    x_ret = np.reshape(output_x, (batch_size, 66, 208, 3))
                    y_ret = np.reshape(output_y, (batch_size))
                    
                    # Reinitialize output lists
                    output_x = []
                    output_y = []
                    yield (x_ret, y_ret)
                
def dnn_model(weights_path = None):
    image_model = Sequential()
    # Normalization layer
    image_model.add(BatchNormalization(input_shape=(66, 208, 3)))
    
    # First convolutional layer
    image_model.add(Convolution2D(24, 5, 5, border_mode='same', init='he_normal', subsample=(2, 2)))
    image_model.add(Activation('relu'))
    #image_model.add(MaxPooling2D(pool_size=(2, 2)))

    # Second convolutional layer
    image_model.add(Convolution2D(36, 5, 5, border_mode='same', init='he_normal', subsample=(2, 2)))
    image_model.add(Activation('relu'))
    #image_model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Third convolutional layer
    image_model.add(Convolution2D(48, 5, 5, border_mode='same', init='he_normal', subsample=(2, 2)))
    image_model.add(Activation('relu'))
    #image_model.add(MaxPooling2D(pool_size=(2, 2)))

    # Fourth convolutional layer
    image_model.add(Convolution2D(64, 3, 3, border_mode='same', init='he_normal', subsample=(2, 2)))
    image_model.add(Activation('relu'))
    #image_model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Fith convolutionla layer
    image_model.add(Convolution2D(64, 3, 3, border_mode='same', init='he_normal', subsample=(2, 2)))
    image_model.add(Activation('relu'))
    #image_model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Flattening
    image_model.add(Flatten())

    # First fully connected layer
    #image_model.add(Dense(200, init='he_normal'))
    #image_model.add(Activation('relu'))
    #image_model.add(Dropout(DROPOUT_RATE))
    
    # Second fully connected layer
    image_model.add(Dense(100, init='he_normal'))
    image_model.add(Activation('relu'))
    image_model.add(Dropout(DROPOUT_RATE))
    
    # Third fully connected layer
    image_model.add(Dense(50, init='he_normal'))
    image_model.add(Activation('relu'))
    image_model.add(Dropout(DROPOUT_RATE))
    
    # Fourth fully connected layer
    image_model.add(Dense(10, init='he_normal'))
    image_model.add(Activation('relu'))
    image_model.add(Dropout(DROPOUT_RATE))

    
    # Final layer
    image_model.add(Dense(1, init='he_normal'))
    image_model.add(Activation('linear'))
    
    if weights_path:
        image_model.load_weights(weights_path)
    
    return(image_model)


if __name__ == "__main__":
    # Generate model
    model = dnn_model()
    # Generate optimizer
    adam = Adam(lr=LEARNING_RATE)
    # Compile model
    model.compile(optimizer=adam, 
                  loss='mse',
                  metrics=['mae', 'mse'])
    
    # checkpoint: to have all eopchs saved
    checkpoint = ModelCheckpoint("model-{epoch:02d}.h5",
                                 monitor='loss',
                                 verbose=1,
                                 save_best_only=False,
                                 mode='max')
    
    callbacks_list = [checkpoint]
    
    # Fit model: epoch is defined to have 5000 samples
    # No validation set
    model.fit_generator(generate_train(BATCH_SIZE),
                        samples_per_epoch=2500, 
                        nb_epoch=EPOCHS,
                        validation_data=None,
                        callbacks=callbacks_list)
    
    # Export model architecture and final weights
    model_json = model.to_json()
    model.save_weights("model.h5")
    with open('model.json', 'w') as f:
        json.dump(model_json, f)
    


