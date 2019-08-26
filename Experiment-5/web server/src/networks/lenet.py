"""Builds the Lenet Model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple
# import tensorflow as tf
# from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda, MaxPooling2D
# from tensorflow.keras.models import Sequential, Model
import keras.backend as K
from keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda, MaxPooling2D
from keras.models import Sequential, Model


def lenet(input_shape : Tuple[int, ...], output_shape : Tuple[int, ...]) -> Model:
    """Creates a lenet model 
    INPUT => CONV => RELU => CONV => RELU => POOL => DROPOUT => Flatten => FC => RELU => DROPOUT => FC
    Args:
    input_shape : shape of the input tensor
    output_shape : shape of the output tensor

    Returns:
    Lenet Model
    """
    num_classes = output_shape[0]
    model = Sequential()
    if len(input_shape) < 3:
        model.add(Lambda(lambda x: K.expand_dims(x, -1), input_shape=input_shape))
        input_shape = (input_shape[0], input_shape[1], 1)
    #Input (28, 28, 1)  -> Output (26, 26, 32)
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    #Input (26, 26, 32) -> Output (24, 24, 64)    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    #Input (24, 24, 64) -> Output (12, 12, 64)    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #Input (24, 24, 64) -> Output (12, 12, 64)    
    model.add(Dropout(0.5))
    #Input (12, 12, 64) -> Output (12*12*64,)    
    model.add(Flatten())
    #Input (12*12*64,)  -> Output (128,)
    model.add(Dense(128, activation='relu'))
    #Input (128,)       -> Output (128,)
    model.add(Dropout(0.5))
    #Input (128,)       -> Output (num_classes,)
    model.add(Dense(num_classes, activation='softmax'))

    return model
