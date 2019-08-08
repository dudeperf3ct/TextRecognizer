"""Builds the CustomCNN Model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple
# import tensorflow as tf
# from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda, MaxPooling2D, LeakyReLU, BatchNormalization
# from tensorflow.keras.models import Sequential, Model
import keras.backend as K
from keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda
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
    #Input (28, 28, 1)  -> Output (26, 26, 32)
    model.add(Conv2D(32, (3,3)))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    #Input (26, 26, 32)  -> Output (24, 24, 32)
    model.add(Conv2D(32, (3,3)))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    #Input (24, 24, 32)  -> Output (22, 22, 64)
    model.add(Conv2D(64, (3,3)))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    #Input (22, 22, 64)  -> Output (10, 10, 64)
    model.add(Conv2D(64, (3,3)))
    model.add(LeakyReLU())
    model.add(MaxPooling2D())
    #Input (10, 10, 64)  -> Output (6400,)
    model.add(Flatten())
    model.add(BatchNormalization())
    #Input (6400,)  -> Output (128,)
    model.add(Dense(128))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    #Input (128,)  -> Output (num_classes,)
    model.add(Dense(num_classes, activation='softmax'))

    return model

# def dummy():
    
# #     # Define the input as a tensor with shape input_shape
#     input_shape = (28, 28)
#     num_classes = 62
#     model = Sequential()
#     if len(input_shape) < 3:
#         model.add(Lambda(lambda x: tf.expand_dims(x, -1), input_shape=input_shape))
#         input_shape = (input_shape[0], input_shape[1], 1)
#     #Input (28, 28, 1)  -> Output (26, 26, 32)
#     model.add(Conv2D(32, (3,3)))
#     model.add(LeakyReLU())
#     model.add(BatchNormalization(axis=1))
#     #Input (26, 26, 32)  -> Output (24, 24, 32)
#     model.add(Conv2D(32, (3,3)))
#     model.add(LeakyReLU())
#     model.add(BatchNormalization(axis=3))
#     #Input (24, 24, 32)  -> Output (22, 22, 64)
#     model.add(Conv2D(64, (3,3)))
#     model.add(LeakyReLU())
#     model.add(BatchNormalization(axis=3))
#     #Input (22, 22, 64)  -> Output (10, 10, 64)
#     model.add(Conv2D(64, (3,3)))
#     model.add(LeakyReLU())
#     model.add(MaxPooling2D())
#     #Input (10, 10, 64)  -> Output (6400,)
#     model.add(Flatten())
#     model.add(BatchNormalization())
#     #Input (6400,)  -> Output (128,)
#     model.add(Dense(128))
#     model.add(LeakyReLU())
#     model.add(BatchNormalization())
#     model.add(Dropout(0.2))
#     #Input (128,)  -> Output (num_classes,)
#     model.add(Dense(num_classes, activation='softmax'))
#     model.summary()
#     return model

# dummy()