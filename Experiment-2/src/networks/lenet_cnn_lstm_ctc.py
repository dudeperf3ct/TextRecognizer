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

    Returns:
    CNN LSTM Model
    """