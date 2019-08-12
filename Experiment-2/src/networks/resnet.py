"""Builds the Resnet Model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple
# import tensorflow as tf
# from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda, GlobalAveragePooling2D, Input
# from tensorflow.keras.layers import MaxPooling2D, ZeroPadding2D, Activation, BatchNormalization, Add
# from tensorflow.keras.models import Sequential, Model
import keras.backend as K
from keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda, GlobalAveragePooling2D, Input
from keras.layers import MaxPooling2D, ZeroPadding2D, Activation, BatchNormalization, Add
from keras.models import Sequential, Model

def _bn_relu(x, bn_name : str = "bn_name", relu_name : str = "relu_name"):
    """Helper to build a BN => RELU block

    Args:
    x: Tensor that is the output of the previous layer in the model.

    """
    norm = BatchNormalization(axis=3, name=bn_name)(x)
    
    return Activation("relu", name=relu_name)(norm)


def _bn_relu_conv(x, 
                  filters : int, kernel_size : int, 
                  strides : int, padding : str,
                  stage : int, block : str) :
    """
    Helper to build a BN => RELU => CONV block

    Args:
    x: Tensor that is the output of the previous layer in the model.

    """
    conv_name = 'conv' + str(stage) + '_' + block 
    bn_name = 'bn' + str(stage) + '_' + block
    relu_name = 'relu' + str(stage) + '_' +  block

    activation = _bn_relu(x, bn_name, relu_name)
    
    x =  Conv2D(filters=filters, kernel_size=kernel_size,
                    strides=strides, padding=padding,
                    name=conv_name)(activation)

    return x

def _conv_bn(x, 
             filters : int, kernel_size : int,
             strides : int, padding : str, 
             stage : int, block : str):

    conv_name = 'conv' + str(stage) + '_' + block 
    bn_name = 'bn' + str(stage) + '_' + block    

    x = Conv2D(filters=filters, kernel_size=kernel_size,
               strides=strides, padding=padding,
               name=conv_name)(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)

    return x

def resnet_block(x,
                 filters: list, kernel_size : list,
                 strides : list, padding : list,
                 stages : list , blocks : list):

    orig_x = x
    x = _bn_relu_conv(x, filters[0], kernel_size[0], strides[0], padding[0], stages[0], blocks[0])
    x = _bn_relu_conv(x, filters[1], kernel_size[1], strides[1], padding[1], stages[1], blocks[1])
    x = _conv_bn(x, filters[2], kernel_size[2], strides[2], padding[2], stages[2], blocks[2])

    x_shortcut = _conv_bn(orig_x, filters[3], kernel_size[3], strides[3], padding[3], stages[3], blocks[3])

    x = Add()([x, x_shortcut])
    x = Activation('relu', name='relu'+str(stages[3]))(x)

    return x


def resnet(input_shape : Tuple[int, ...], output_shape : Tuple[int, ...]) -> Model:
#     """Creates a resnet model 
#     Stage-1
#     INPUT => CONV => BN => RELU
    
#     Stage-2
#     RESNET_BLOCK
    
#     Stage-3
#     RESNET_BLOCK    

#     Stage-4
#     DROPOUT => Flatten => FC => RELU => DROPOUT => FC
    
#     Args:
#     input_shape : shape of the input tensor
#     num_classes : number of classes

#     Returns:
#     Resnet Model
#     """
    num_classes = output_shape[0]

    X_input = Input(input_shape)
    if len(input_shape) < 3:
        X = Lambda(lambda x: K.expand_dims(x, -1), input_shape=(input_shape[0], input_shape[1]))(X_input) 
    # Stage 1
    X = ZeroPadding2D((2, 2))(X)
    X = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv1')(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)

    # Stage 2
    #Input (16, 16, 64)       -> Output (16, 16, 128)
    filters = [64, 64, 128, 128]
    kernels = [1, 3, 1, 1]
    strides = [1, 1, 1, 1]
    paddings = ['valid', 'same', 'valid', 'valid'] 
    stages = [2, 2, 2, 2] 
    blocks = ['2a', '2b', '2c', '2']
    X = resnet_block(X, filters, kernels, strides, paddings, stages, blocks)

    # Stage 3
    #Input (16, 16, 128)       -> Output (8, 8, 512)
    filters = [256, 256, 512, 512]
    kernels = [1, 3, 1, 1]
    strides = [2, 1, 1, 2]
    padding = ['valid', 'same', 'valid', 'valid'] 
    stages = [3, 3, 3, 3] 
    blocks = ['3a', '3b', '3c', '3']
    X = resnet_block(X, filters, kernels, strides, paddings, stages, blocks)

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

    return model

# def dummy():
    
#     # Define the input as a tensor with shape input_shape
#     input_shape = (28, 28)

#     X_input = Input(input_shape)
#     if len(input_shape) < 3:
#         X = Lambda(lambda x: tf.expand_dims(x, -1), input_shape=(input_shape[0], input_shape[1]))(X_input) 
#     # Stage 1
#     X = ZeroPadding2D((2, 2))(X)
#     X = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv1')(X)
#     X = BatchNormalization(axis=3, name='bn_conv1')(X)
#     X = Activation('relu')(X)
#     X = MaxPooling2D(pool_size=(2, 2))(X)

#     # Stage 2
#     filters = [64, 64, 128, 128]
#     kernels = [1, 3, 1, 1]
#     strides = [1, 1, 1, 1]
#     paddings = ['valid', 'same', 'valid', 'valid'] 
#     stages = [2, 2, 2, 2] 
#     blocks = ['2a', '2b', '2c', '2']
#     X = resnet_block(X, filters, kernels, strides, paddings, stages, blocks)

#     # Stage 3
#     filters = [256, 256, 512, 512]
#     kernels = [1, 3, 1, 1]
#     strides = [2, 1, 1, 2]
#     padding = ['valid', 'same', 'valid', 'valid'] 
#     stages = [3, 3, 3, 3] 
#     blocks = ['3a', '3b', '3c', '3']
#     X = resnet_block(X, filters, kernels, strides, paddings, stages, blocks)

#     X = MaxPooling2D(pool_size=(2, 2))(X)
#     #Input () -> Output ()    
#     X = Dropout(0.2)(X)
#     #Input () -> Output (,)    
#     X = GlobalAveragePooling2D()(X)
#     #Input (,)  -> Output (128,)
#     X = Dense(128, activation='relu')(X)
#     #Input (128,)       -> Output (128,)
#     X = Dropout(0.2)(X)
#     #Input (128,)       -> Output (num_classes,)
#     X = Dense(62, activation='softmax')(X)

#     model = Model(inputs=X_input, outputs=X)
#     model.summary()

#     return model

# dummy()