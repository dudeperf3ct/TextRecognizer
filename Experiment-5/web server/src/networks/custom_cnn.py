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
from keras.layers import MaxPooling2D, LeakyReLU, BatchNormalization, Reshape
from keras.models import Sequential, Model


def customcnn(input_shape : Tuple[int, ...],
              output_shape : Tuple[int, ...],
              window_width: float = 16,
              window_stride: float = 8) -> Model:
    """Creates a custom cnn model 
    
    Args:
    input_shape : shape of the input tensor
    num_classes : number of classes
    window_width : width of sliding window
    window_stride : stride of sliding window
    Returns:
    CustomCNN Model
    """
    image_height, image_width = input_shape    # 28, 952
    output_length, num_classes = output_shape  # 34, 64

    model = Sequential()
    if len(input_shape) < 3:
        model.add(Lambda(lambda x: K.expand_dims(x, -1), input_shape=input_shape))
    input_shape = (input_shape[0], input_shape[1], 1)

    #Input (28, 952, 1)  -> Output (26, 950, 32)
    model.add(Conv2D(32, (3,3)))
    model.add(ReLU())
    model.add(BatchNormalization())

    #Input (26, 950, 32)  -> Output (24, 948, 64)
    model.add(Conv2D(64, (3,3)))
    model.add(ReLU())
    model.add(BatchNormalization())
    #Input (24, 948, 64)  -> Output (11, 473, 64)
    model.add(Conv2D(64, (3,3)))
    model.add(ReLU())
    model.add(MaxPooling2D())

    new_height = image_height // 2 - 3         #11
    new_width = image_width // 2 - 3           #473
    new_window_width = window_width // 2 - 3   #5
    new_window_stride = window_stride // 2     #4
    num_windows = int((new_width - new_window_width) / new_window_stride) + 1  #118

    # Input (11, 473, 64) -> Output (1, 118, 128)    
    model.add(Conv2D(128, (new_height, new_window_width), strides=(1, new_window_stride), activation='relu'))
    # Input (11, 473, 64) -> Output (1, 118, 128)    
    model.add(Dropout(0.2))

    # Input (1, 118, 128) -> Output (118, 128, 1)
    model.add(Reshape((num_windows, 128, 1)))

    width = int(num_windows / output_length)  #3

    # Input (118, 128, 1) -> Output (39, 1, 64)
    model.add(Conv2D(num_classes, (width, 128), strides=(width, 1), activation='softmax'))

    # Input (39, 1, 64) -> Output (39, 64)
    model.add(Lambda(lambda x: K.squeeze(x, 2)))

    # Input (39, 1, 64) -> Output (output_length, 64)
    # Since we floor'd the calculation of width, we might have too many items in the sequence. Take only output_length.
    model.add(Lambda(lambda x: x[:, :output_length, :]))

    model.summary()

    return model

# def dummy():
#     input_shape = (28, 952)
#     output_shape = (34, 64)
#     window_width = 16
#     window_stride = 8
#     image_height, image_width = input_shape    # 28, 952
#     output_length, num_classes = output_shape  # 34, 64

#     model = Sequential()
#     if len(input_shape) < 3:
#         model.add(Lambda(lambda x: K.expand_dims(x, -1), input_shape=input_shape))
#     input_shape = (input_shape[0], input_shape[1], 1)

#     #Input (28, 952, 1)  -> Output (26, 950, 32)
#     model.add(Conv2D(32, (3,3)))
#     model.add(ReLU())
#     model.add(BatchNormalization())

#     #Input (26, 950, 32)  -> Output (24, 948, 64)
#     model.add(Conv2D(64, (3,3)))
#     model.add(ReLU())
#     model.add(BatchNormalization())
#     #Input (24, 948, 64)  -> Output (11, 473, 64)
#     model.add(Conv2D(64, (3,3)))
#     model.add(ReLU())
#     model.add(MaxPooling2D())

#     new_height = image_height // 2 - 3         #11
#     new_width = image_width // 2 - 3           #473
#     new_window_width = window_width // 2 - 3   #5
#     new_window_stride = window_stride // 2     #4
#     num_windows = int((new_width - new_window_width) / new_window_stride) + 1  #118

#     # Input (11, 473, 64) -> Output (1, 118, 128)    
#     model.add(Conv2D(128, (new_height, new_window_width), strides=(1, new_window_stride), activation='relu'))
#     # Input (11, 473, 64) -> Output (1, 118, 128)    
#     model.add(Dropout(0.2))

#     # Input (1, 118, 128) -> Output (118, 128, 1)
#     model.add(Reshape((num_windows, 128, 1)))

#     width = int(num_windows / output_length)  #3

#     print (new_height, new_width)
#     print (new_window_width, new_window_stride)
#     print (num_windows, width)

#     # Input (118, 128, 1) -> Output (39, 1, 64)
#     model.add(Conv2D(num_classes, (width, 128), strides=(width, 1), activation='softmax'))

#     # Input (39, 1, 64) -> Output (39, 64)
#     model.add(Lambda(lambda x: K.squeeze(x, 2)))

#     # Input (39, 1, 64) -> Output (output_length, 64)
#     # Since we floor'd the calculation of width, we might have too many items in the sequence. Take only output_length.
#     model.add(Lambda(lambda x: x[:, :output_length, :]))

#     model.summary()

#     return model

# dummy()    