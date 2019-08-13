"""Builds the CNN-LSTM Model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple
# import tensorflow as tf
# from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda, MaxPooling2D
# from tensorflow.keras.models import Sequential, Model
import keras.backend as K
from keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda, MaxPooling2D, Reshape, LSTM
from keras.models import Sequential, Model


def lenetcnnlstm(input_shape : Tuple[int, ...], 
                output_shape : Tuple[int, ...],
                window_width: float = 16,
                window_stride: float = 8) -> Model:
    """Creates a lenet cnn lstm model 
    INPUT => CONV => RELU => CONV => RELU => POOL => DROPOUT => Flatten => FC => RELU => DROPOUT => FC
    Args:
    input_shape : shape of the input tensor
    output_shape : shape of the output tensor
    window_width : width of sliding window
    window_stride : stride of sliding window
    Returns:
    CNN LSTM Model
    """

    image_height, image_width = input_shape    # 28, 952
    output_length, num_classes = output_shape  # 34, 64
    num_windows = int((image_width - window_width) / window_stride) + 1 #118

    if num_windows < output_length:
        raise ValueError(f'Window width/stride need to generate >= {output_length} windows (currently {num_windows})')

    # Input (28, 952) -> Output (28, 952, 1)
    image_input = Input(shape=input_shape)
    image_reshaped = Reshape((image_height, image_width, 1))(image_input)

    image_patches = Lambda(
        slide_window,
        arguments={'window_width': window_width, 'window_stride': window_stride}
    )(image_reshaped)
    # (118, 28, 16, 1)