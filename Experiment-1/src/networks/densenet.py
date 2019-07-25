"""Builds the Densenet Model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Input, ZeroPadding2D, Activation, BatchNormalization, Add
from tensorflow.keras.models import Sequential, Model





def densenet(input_shape : Tuple[int, ...], output_shape : Tuple[int, ...]) -> Model:
#     """Creates a resnet model 
#     Stage-1
#     INPUT => CONV => BN => RELU
    
#     Stage-2
#     DENSENET_BLOCK
    
#     Stage-3
#     DENSENET_BLOCK    

#     Stage-4
#     DROPOUT => Flatten => FC => RELU => DROPOUT => FC
    
#     Args:
#     input_shape : shape of the input tensor
#     num_classes : number of classes

#     Returns:
#     Resnet Model
#     """
    num_classes = output_shape[0]

    if len(input_shape) < 3:
        model.add(Lambda(lambda x: tf.expand_dims(x, -1), input_shape=input_shape))
        input_shape = (input_shape[0], input_shape[1], 1)
    
    # Define the input as a tensor with shape input_shape
    X_input = Input((28, 28 , 1))
    # Stage 1
    X = ZeroPadding2D((2, 2))(X_input)
    X = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv1')(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)

    # Stage 2
    #Input (16, 16, 64)       -> Output (16, 16, 128)


    # Stage 3
    #Input (16, 16, 128)       -> Output (8, 8, 512)


    #Input (8, 8, 512) -> Output (4, 4, 512) 
    X = MaxPooling2D(pool_size=(2, 2))(X)
    #Input (4, 4, 512) -> Output (4, 4, 512)  
    X = Dropout(0.2)(X)
    #Input (4, 4, 512)  -> Output (512,)   
    X = GlobalAveragePooling2D()(X)
    #Input (512,)       -> Output (128,)
    X = Dense(128, activation='relu')(X)
    #Input (128,)       -> Output (128,)
    X = Dropout(0.2)(X)
    #Input (128,)       -> Output (num_classes,)
    X = Dense(num_classes, activation='softmax')(X)

    model = Model(inputs=X_input, outputs=X)
    model.summary()

    return model

# def dummy():
    
#     # Define the input as a tensor with shape input_shape
#     X_input = Input((28, 28 , 1))
#     # Stage 1
#     X = ZeroPadding2D((2, 2))(X_input)
#     X = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv1')(X)
#     X = BatchNormalization(axis=3, name='bn_conv1')(X)
#     X = Activation('relu')(X)
#     X = MaxPooling2D(pool_size=(2, 2))(X)

#     model = Model(inputs=X_input, outputs=X)
#     model.summary()

#     return model

# dummy()