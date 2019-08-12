"""Builds the CNN Model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple
# import tensorflow as tf
# from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda, MaxPooling2D
# from tensorflow.keras.models import Sequential, Model
import keras.backend as K
from keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda, MaxPooling2D, Reshape
from keras.models import Sequential, Model


def cnn(input_shape : Tuple[int, ...], 
        output_shape : Tuple[int, ...],
        window_width: float = 16,
        window_stride: float = 8) -> Model:
    """Creates a lenet model 
    INPUT => CONV => RELU => CONV => RELU => POOL => DROPOUT => Flatten => FC => RELU => DROPOUT => FC
    Args:
    input_shape : shape of the input tensor
    output_shape : shape of the output tensor

    Returns:
    CNN Model
    """

    image_height, image_width = input_shape    # 28, 
    output_length, num_classes = output_shape  # 34, 
 
    model = Sequential()
    # Input (28, 952) -> Output (28, 952, 1)
    model.add(Reshape((image_height, image_width, 1), input_shape=input_shape))
    # Input (28, 952, 1) -> Output (26, 950, 1)    
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    # Input (26, 950, 32) -> Output (24, 948, 64)
    model.add(Conv2D(64, (3, 3), activation='relu'))
    # Input (24, 948, 64) -> Output (12, 474, 64)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Input (12, 474, 64) -> Output (12, 474, 64)
    model.add(Dropout(0.2))

    new_height = image_height // 2 - 2
    new_width = image_width // 2 - 2
    new_window_width = window_width // 2 - 2
    new_window_stride = window_stride // 2
    num_windows = int((new_width - new_window_width) / new_window_stride) + 1
    
    # Input (12, 474, 64) -> Output (1, 118, 128)    
    model.add(Conv2D(128, (new_height, new_window_width), strides=(1, new_window_stride), activation='relu'))
    # Input (12, 474, 64) -> Output (1, 118, 128)    
    model.add(Dropout(0.2))
    
    # Input (1, 118, 128) -> Output (118, 128, 1)
    model.add(Reshape((num_windows, 128, 1)))

    width = int(num_windows / output_length)

    # Input (118, 128, 1) -> Output (39, 1, 64)
    model.add(Conv2D(num_classes, (width, 128), strides=(width, 1), activation='softmax'))

    # Input (39, 1, 64) -> Output (39, 64)
    model.add(Lambda(lambda x: K.squeeze(x, 2)))

    # Input (39, 1, 64) -> Output (39, 64)
    # Since we floor'd the calculation of width, we might have too many items in the sequence. Take only output_length.
    model.add(Lambda(lambda x: x[:, :output_length, :]))

    return model

# def dummy():
#     input_shape = (28, 952)
#     output_shape = (34, 64)
#     window_width = 16
#     window_stride = 8
#     image_height, image_width = input_shape    # 28, 
#     output_length, num_classes = output_shape  # 34, 

#     model = Sequential()
#     model.add(Reshape((image_height, image_width, 1), input_shape=input_shape))
#     model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
#     model.add(Conv2D(64, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.2))
#     # (12, 474, 64)

#     new_height = image_height // 2 - 2         #12
#     new_width = image_width // 2 - 2           #474
#     new_window_width = window_width // 2 - 2   #4
#     new_window_stride = window_stride // 2     #6
#     num_windows = int((new_width - new_window_width) / new_window_stride) + 1 #118
#     width = int(num_windows / output_length)   #3

#     print (new_height, new_width)
#     print (new_window_stride, new_window_width)
#     print (num_windows, width)

#     model.add(Conv2D(128, (new_height, new_window_width), strides=(1, new_window_stride), activation='relu'))
#     model.add(Dropout(0.2))
#     # (num_windows, 128)
#     model.add(Reshape((num_windows, 128, 1)))
#     model.add(Conv2D(num_classes, (width, 128), strides=(width, 1), activation='softmax'))
#     model.add(Lambda(lambda x: K.squeeze(x, 2)))
#     model.add(Lambda(lambda x: x[:, :output_length, :]))

#     model.summary()

# dummy()