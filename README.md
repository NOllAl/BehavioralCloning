# BehavioralCloning

In this project, a deep convolutional neural network predicting the steering angles of a car is trained. 
The model is defined and trained using the Keras API in Python. The steps are as follows:

1. Collect training data by running the simulator and steering the car
2. Preprocess the training data
3. Define a deep convolutional neural network using Keras
4. Train and validate the model
5. Test the model by running the model in autonomous mode

We now proceed with a more detailed description:

# Data collection

The dataset used as training set comes from playing the simulator. The simulator has a recording mode which reocrds the throttle and the steering angle set by the player. Furthermore, it records images as if three cameras were mounted at the front of the car

1. One camera is placed at the center of the car
2. A second camera is placed to the left of the center
3. The thrid camera is placed to the right of the center of the windshield

The usage of these additional cameras will be explained below.

The simulator was played so long until 40000 images have been collected.

# Preprocessing

The data are preprocessed as follows. First of all, the images are cut down from 320 x 160 to 208 x 66 by first selecting
only the most relevant parts in the y-direction: we take only the pixel from 32:135 to restrict to the *road*. 
Then, in order to decrease the training time, we resize the image to 208 x 66. Furthermore, the resized images are then normalized so that the pixel values lie between -1 and 1.

This kind of preprocessing is used in both training and in running the simulator. The corresponding function can be found as `preproc_image` in the python file `preproc.py`. 

# Deep neural net

## Model architecture

The following model architecture was used:

1. Convolutional layer of depth 24 with a filter size of 5x5 and subsample of 2x2
2. Convolutional layer of depth 36 with a filter size of 5x5 and subsample of 2x2
3. Convolutional layer of depth 48 with a filter size of 5x5 and subsample of 2x2
4. Convolutoinal layer of depth 64 with a filter size of 5x5 and subsample of 2x2
5. Convolutional layer of depth 64 with a filter size of 5x5 and subsample of 2x2
6. Flattening layer
7. Fully connected layer of size 100
8. Fully connected layer of size 50
9. Fully connected layer of size 10
10. Output layer of size 1

All activation functions are 'relus'. After each fully connected layer, we add a dropout with a rate of 0.35 to reduce overfitting.

## Generator

The Keras model is trained using a custom Python generator returning a preprocessed image and a steering angle. 
This generator works as follows: it shuffles the training data by only considering each 1000th image 
(thus, it produces a kind of bootstrapped version of the training set). Furthremore, with a probability of 1/3 each, it 
reads either the left, center, or right image described above. If the left image is selected, the steering angle is shifted
by +0.25. If the right image is selected, the steering angle is shifted by -0.25. Otherwise the steering angle is kept as is.

As a next step, in order to artificially increase the size of the training set and in order to reduce overfitting, the image is translated by a random amount between -100 and 100 in the x-direction. Together with the shift, the steering angle is also shifted by an appropriate amount (namely the amount of shifting times the offset of 0.25). This kind of translation will also resulte in the car learning to correct the steering angle better.

Next, with a probability of 0.5 the image is flipped along a vertical axis and the steering angle is multiplied by -1. 
This is done in order to augment the training set and to reduce overfitting. Basically it increases the training set by a factor of 2.

The batch size of the generator is customizable (and set to 100 in training).

# Training

In order to train the model, an Adam optimizer with a learning rate of 0.00006 is used. This learning rate was found by 
cross validation. During the training process, after each epoch, the intermediate results are saved away, so that 
the model after each epoch can be used for testing whether the car drives well. This is a crucial step: the number of epochs turns out to be an extremely important parameter:

If the model runs over too few epochs, it fails to steer strongly enough. If it traing too long, it becomes very certain, even for very large angles. This is a problem since it leads to the car being very unstable. It was found that the cars drives best when it is trained over 27 epochs using an epoch size of 2500, i.e. effectively training with 67500 images.

This is an example of early stopping to avoid overfitting. Another option might have been to include regularization.


## Validation

A validation generator was generated, similar to the training generator described above. This can be set in the `model.py` file by passing it to the `valdation_data` argument in the `model.fit_generator()` call. This argument is set to `None` in the final model run: in order to enlarge the training data further, after making sure that the model does not overfit, all training data are passed to the training.

Another type of valdiation heavily employed in this project was to run the simulator many times and looking for the kind of errors the car makes.

# Testing in autonomous mode

The final model can be run with `python drive.py model.json` and then launching the simulator in autonomous mode. It is found that the car can drive several laps reliably.
