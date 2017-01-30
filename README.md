# BehavioralCloning

In this project, a deep convolutional neural network predicting the steering angles of a car is trained. 
The model is defined and trained using the Keras API in Python. We now proceed with a more detailed project description.

## Data

The dataset used as training set comes from playing a computer game which records the steering angle of the player. 
Furthermore, each image in the game is saved as `.jpg` file. In addition to an image captured at the center of the car, there 
are two additional images translated to the left and translated to the right, respectively.

## Preprocessing

The data are preprocessed as follows. First of all, the images are cut down from 320 x 160 to 208 x 66 by first selecting
only the most relevant parts in the y-direction: we take only the pixel from 32:135 to restrict to the *road*. 
Then, in order to decrease the training time, we resize the image to 208 x 66. 

## Model architecture

The following model architecture was used:

1. Convolutional layer of depth 24 with a filter size of 5x5 and subsample of 2x2
2. Convolutional layer of depth 36 with a filter size of 5x5 and subsample of 2x2
3. Convolutional layer of depth 48 with a filter size of 5x5 and subsample of 2x2
4. Convolutoinal layer of depth 64 with a filter size of 5x5 and subsample of 2x2
5. Convolutional layer of depth 64 with a filter size of 5x5 and subsample of 2x2
6. Flattening layer
7. Fully connected layer of size 200 
8. Fully connected layer of size 100
9. Fully connected layer of size 50
10. Fully connected layer of size 10
11. Output layer of size 1

All activation functions are relus. After each fully connected layer, we add a dropout with a rate of 0.5.

## Generator

The Keras model is trained using a custom Python generator returning a preprocessed image and a steering angle. 
This generator works as follows: it shuffles the training data by only considering each 1000th image 
(thus, it produces a kind of bootstrapped version of the training set). Furthremore, with a probability of 1/3 each, it 
reads either the left, center, or right image described above. If the left image is selected, the steering angle is shifted
by +0.2. If the right image is selected, the steering angle is shifted by -0.2. Otherwise the steering angle is kept as is.

Next, with a probability of 0.5 the image is flipped along a vertical axis and the steering angle is multiplied by -1. 
This is done in order to augment the training set. 

The batch size of the generator is customizable (and set to 100 in training).

## Training

In order to train the model, an Adam optimizer with a learning rate of 0.00006 is used. This learning rate was found by 
cross validation. During the training process, after each epoch, the intermediate results are saved away, so that 
the model after each epoch can be used for testing whether the car drives well.

## Validation

The model is not directly evaluated using the error metric ('MSE'): the goal is not to predict the steering angle
as accurately as possible, but to have a well driving car. So, this is how the model was evaluated: the models were 
simply put to use in the simulator. 
