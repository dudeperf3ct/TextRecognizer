"""Builds the Resnet Model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple
# import tensorflow as tf
# from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda, GlobalAveragePooling2D, Input
# from tensorflow.keras.layers import MaxPooling2D, ZeroPadding2D, Activation, BatchNormalization, Add, Reshape
# from tensorflow.keras.models import Sequential, Model
import keras.backend as K
from keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda, GlobalAveragePooling2D, Input
from keras.layers import MaxPooling2D, ZeroPadding2D, Activation, BatchNormalization, Add, Reshape
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


def resnetcnn(input_shape : Tuple[int, ...],
            output_shape : Tuple[int, ...],
            window_width: float = 16,
            window_stride: float = 8) -> Model:
    """Creates a resnet model 
    Stage-1
    INPUT => CONV => BN => RELU
    
    Stage-2
    RESNET_BLOCK
    
    Stage-3
    RESNET_BLOCK    

    Stage-4
    DROPOUT => Flatten => FC => RELU => DROPOUT => FC
    
    Args:
    input_shape : shape of the input tensor
    num_classes : number of classes
    window_width : width of sliding window
    window_stride : stride of sliding window
    Returns:
    Resnet Model
    """
    image_height, image_width = input_shape    # 28, 952
    output_length, num_classes = output_shape  # 34, 64

    X_input = Input(input_shape)
    if len(input_shape) < 3:
        X = Lambda(lambda x: K.expand_dims(x, -1), input_shape=(input_shape[0], input_shape[1]))(X_input) 
    # Stage 1
    # Input (28, 952, 1) -> Output (32, 956, 1)
    X = ZeroPadding2D((2, 2))(X)
    # Input (32, 956, 1) -> Output (32, 956, 64)
    X = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv1')(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    # Input (32, 956, 64) -> Output (16, 478, 64)    
    X = MaxPooling2D(pool_size=(2, 2))(X)

    # Stage 2
    #Input (16, 478, 64)   -> Output (16, 478, 128)
    filters = [64, 64, 128, 128]
    kernels = [1, 3, 1, 1]
    strides = [1, 1, 1, 1]
    paddings = ['valid', 'same', 'valid', 'valid'] 
    stages = [2, 2, 2, 2] 
    blocks = ['2a', '2b', '2c', '2']
    X = resnet_block(X, filters, kernels, strides, paddings, stages, blocks)

    # Stage 3
    #Input (16, 478, 128)   -> Output (8, 239, 512)
    filters = [256, 256, 512, 512]
    kernels = [1, 3, 1, 1]
    strides = [2, 1, 1, 2]
    padding = ['valid', 'same', 'valid', 'valid'] 
    stages = [3, 3, 3, 3] 
    blocks = ['3a', '3b', '3c', '3']
    X = resnet_block(X, filters, kernels, strides, paddings, stages, blocks)

    #Input (8, 239, 512)   -> Output (4, 119, 512)  
    X = MaxPooling2D(pool_size=(2, 2))(X)
    X = Dropout(0.2)(X)
 
    new_height = (image_height + 4) // 8        #4
    new_width = (image_width + 4) // 8          #119
    new_window_width = (window_width + 4) // 8  #2
    new_window_stride = window_stride // 4    #2
    num_windows = int((new_width - new_window_width) / new_window_stride) + 1  #59
    
    # Input (4, 119, 64) -> Output (1, 59, 128)    
    X = Conv2D(128, (new_height, new_window_width), strides=(1, new_window_stride), activation='relu')(X)
    # Input (1, 59, 128) -> Output (1, 59, 128)  
    X = Dropout(0.2)(X)
    
    # Input (1, 59, 128) -> Output (59, 128, 1)
    X = Reshape((num_windows, 128, 1))(X)

    width = int(num_windows / output_length)  #

    # Input (59, 128, 1) -> Output (59, 1, 64)
    X = Conv2D(num_classes, (width, 128), strides=(width, 1), activation='softmax')(X)

    # Input (59, 1, 64) -> Output (59, 64)
    X = Lambda(lambda x: K.squeeze(x, 2))(X)

    # Input (59, 64) -> Output (39, 64)
    # Since we floor'd the calculation of width, we might have too many items in the sequence. Take only output_length.
    outputs = Lambda(lambda x: x[:, :output_length, :])(X)

    model = Model(inputs=X_input, outputs=outputs)

    return model

# def dummy():
#     input_shape = (28, 952)
#     output_shape = (34, 64)
#     image_height, image_width = input_shape    # 28, 952
#     output_length, num_classes = output_shape  # 34, 64
#     window_width = 16
#     window_stride = 8

#     X_input = Input(input_shape)
#     if len(input_shape) < 3:
#         X = Lambda(lambda x: K.expand_dims(x, -1), input_shape=(input_shape[0], input_shape[1]))(X_input) 
#     # Stage 1
#     # Input (28, 952, 1) -> Output (32, 956, 1)
#     X = ZeroPadding2D((2, 2))(X)
#     # Input (32, 956, 1) -> Output (32, 956, 64)
#     X = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv1')(X)
#     X = BatchNormalization(axis=3, name='bn_conv1')(X)
#     X = Activation('relu')(X)
#     # Input (32, 956, 64) -> Output (16, 478, 64)    
#     X = MaxPooling2D(pool_size=(2, 2))(X)

#     # Stage 2
#     #Input (16, 478, 64)   -> Output (16, 478, 128)
#     filters = [64, 64, 128, 128]
#     kernels = [1, 3, 1, 1]
#     strides = [1, 1, 1, 1]
#     paddings = ['valid', 'same', 'valid', 'valid'] 
#     stages = [2, 2, 2, 2] 
#     blocks = ['2a', '2b', '2c', '2']
#     X = resnet_block(X, filters, kernels, strides, paddings, stages, blocks)

#     # Stage 3
#     #Input (16, 478, 128)   -> Output (8, 239, 512)
#     filters = [256, 256, 512, 512]
#     kernels = [1, 3, 1, 1]
#     strides = [2, 1, 1, 2]
#     padding = ['valid', 'same', 'valid', 'valid'] 
#     stages = [3, 3, 3, 3] 
#     blocks = ['3a', '3b', '3c', '3']
#     X = resnet_block(X, filters, kernels, strides, paddings, stages, blocks)

#     #Input (8, 239, 512)   -> Output (4, 119, 512)  
#     X = MaxPooling2D(pool_size=(2, 2))(X)
#     X = Dropout(0.2)(X)
 
#     new_height = (image_height + 4) // 8        #4
#     new_width = (image_width + 4) // 8          #119
#     new_window_width = (window_width + 4) // 8  #2
#     new_window_stride = window_stride // 4    #2
#     num_windows = int((new_width - new_window_width) / new_window_stride) + 1  #58
    
#     # Input (4, 119, 64) -> Output (1, 28, 128)    
#     X = Conv2D(128, (new_height, new_window_width), strides=(1, new_window_stride), activation='relu')(X)
#     # Input () -> Output ()    
#     X = Dropout(0.2)(X)
    
#     # Input (1, 28, 128) -> Output (28, 128, 1)
#     X = Reshape((num_windows, 128, 1))(X)

#     width = int(num_windows / output_length)  #

#     print (new_height, new_width)
#     print (new_window_width, new_window_stride)
#     print (num_windows, width)

#     # Input (118, 128, 1) -> Output (39, 1, 64)
#     X = Conv2D(num_classes, (width, 128), strides=(width, 1), activation='softmax')(X)

#     # Input (39, 1, 64) -> Output (, 64)
#     X = Lambda(lambda x: K.squeeze(x, 2))(X)

#     # Input () -> Output (output_length, 64)
#     # Since we floor'd the calculation of width, we might have too many items in the sequence. Take only output_length.
#     outputs = Lambda(lambda x: x[:, :output_length, :])(X)

#     model = Model(inputs=X_input, outputs=outputs)

#     model.summary()

#     return model

# dummy()