"""Builds the Custom fully CNN Model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple
# import tensorflow as tf
# from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda, MaxPooling2D, LeakyReLU, BatchNormalization
# from tensorflow.keras.models import Sequential, Model
import keras.backend as K
from keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda, ReLU, Input
from keras.layers import MaxPooling2D, LeakyReLU, BatchNormalization
from keras.models import Sequential, Model


def cnn_layer(x, num_filter, kernel_size, dilation):
    x = Conv2D(num_filter, kernel_size, dilation_rate=dilation, padding='same')(x)
    x = ReLU()(x)
    x = BatchNormalization()(x)
    return x

def cnn_pool_layer(x, num_filter, kernel_size, dilation):
    x = Conv2D(num_filter, kernel_size, dilation_rate=dilation, padding='same')(x)
    x = ReLU()(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)
    return x

def customFCN(input_shape : Tuple[int, ...], output_shape : Tuple[int, ...]) -> Model:
    """Creates a custom fully cnn model 
    
    Args:
    input_shape : shape of the input tensor
    output_shape : shape of the output tensor

    Returns:
    custom fcn Model
    """
    num_classes = output_shape[-1] #3

    input_image = Input((None, None))
    x = Lambda(lambda x: K.expand_dims(x, axis=-1))(input_image)

    x = cnn_layer(x, 32, 7, 3)

    x = cnn_pool_layer(x, 64, 7, 3)

    x = cnn_pool_layer(x, 128, 7, 7)

    output = Conv2D(num_classes, (1, 1), dilation_rate=(1, 1), padding='same', activation='softmax')(x)
    
    model = Model(inputs=input_image, outputs=output)

    return model



# def dummy():
#     num_classes = 3
#     input_image = Input((None, None))
#     x = Lambda(lambda x: K.expand_dims(x, axis=-1))(input_image)
#     x = cnn_layer(x, 32, 7, 3)
#     x = cnn_pool_layer(x, 64, 7, 3)
#     x = cnn_pool_layer(x, 128, 7, 7)
#     output = Conv2D(num_classes, (1, 1), dilation_rate=(1, 1), padding='same', activation='softmax')(x)
#     model = Model(inputs=input_image, outputs=output)
#     model.summary()

# dummy()
