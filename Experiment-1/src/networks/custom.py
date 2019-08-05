"""Builds the Lenet Model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple
#import tensorflow as tf
#from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda, MaxPooling2D
#from tensorflow.keras.models import Sequential, Model
import keras.backend as K
from keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda
from keras.layers import MaxPooling2D, LeakyReLU, BatchNormalization
from keras.models import Sequential, Model


def customCNN(input_shape : Tuple[int, ...], output_shape : Tuple[int, ...]) -> Model:
    """Creates a lenet model 
    INPUT => CONV => RELU => CONV => RELU => POOL => DROPOUT => Flatten => FC => RELU => DROPOUT => FC
    Args:
    input_shape : shape of the input tensor
    num_classes : number of classes

    Returns:
    Lenet Model
    """
    num_classes = output_shape[0]
    model = Sequential()
    if len(input_shape) < 3:
        model.add(Lambda(lambda x: K.expand_dims(x, -1), input_shape=input_shape))
        input_shape = (input_shape[0], input_shape[1], 1)
    #Input (28, 28, 1)  -> Output (26, 26, 32)
    model.add(Conv2D(32, (3,3)))
    model.add(LeakyReLU())
    model.add(BatchNormalization(axis=1))
    #Input (26, 26, 32)  -> Output (24, 24, 32)
    model.add(Conv2D(32, (3,3)))
    model.add(LeakyReLU())
    model.add(MaxPooling2D())
    model.add(BatchNormalization(axis=1))
    #Input (24, 24, 32)  -> Output (22, 22, 64)
    model.add(Conv2D(64, (3,3)))
    model.add(LeakyReLU())
    model.add(BatchNormalization(axis=1))
    #Input (22, 22, 64)  -> Output (20, 20, 64)
    model.add(Conv2D(64, (3,3)))
    model.add(LeakyReLU())
    model.add(MaxPooling2D())
    #Input (20, 20, 64)  -> Output (25600,)
    model.add(Flatten())
    model.add(BatchNormalization())
    #Input (25600,)  -> Output (512,)
    model.add(Dense(512))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    #Input (512,)  -> Output (num_classes,)
    model.add(Dense(num_classes, activation='softmax'))

    return model