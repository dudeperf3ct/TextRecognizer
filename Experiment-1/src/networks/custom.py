"""Builds the CustomCNN Model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple
# import tensorflow as tf
# from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda, MaxPooling2D, LeakyReLU, BatchNormalization
# from tensorflow.keras.models import Sequential, Model
import keras.backend as K
from keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda, ReLU
from keras.layers import MaxPooling2D, LeakyReLU, BatchNormalization
from keras.models import Sequential, Model


def customCNN(input_shape : Tuple[int, ...], output_shape : Tuple[int, ...]) -> Model:
    """Creates a custom cnn model 
    
    Args:
    input_shape : shape of the input tensor
    num_classes : number of classes

    Returns:
    CustomCNN Model
    """
    num_classes = output_shape[0]
    model = Sequential()
    if len(input_shape) < 3:
        model.add(Lambda(lambda x: K.expand_dims(x, -1), input_shape=input_shape))
        input_shape = (input_shape[0], input_shape[1], 1)
    #Input (28, 28, 1)  -> Output (13, 13, 32)
    model.add(Conv2D(32, (3,3)))
    model.add(ReLU())
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    #Input (13, 13, 32)  -> Output (11, 11, 64)
    model.add(Conv2D(64, (3,3)))
    model.add(ReLU())
    model.add(BatchNormalization())
    #Input (11, 11, 64)  -> Output (4, 4, 64)
    model.add(Conv2D(64, (3,3)))
    model.add(ReLU())
    model.add(MaxPooling2D())
    #Input (4, 4, 64)  -> Output (1024,)
    model.add(Flatten())
    model.add(BatchNormalization())
    #Input (1024,)  -> Output (128,)
    model.add(Dense(128))
    model.add(ReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    #Input (128,)  -> Output (num_classes,)
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    return model