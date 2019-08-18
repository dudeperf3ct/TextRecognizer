"""Builds the CNN-LSTM Model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.networks.sliding import slide_window
from src.networks.ctc import ctc_decode
from src.networks.lenet import lenet
from src.networks.resnet import resnet
from src.networks.custom import customCNN
from tensorflow.python.client import device_lib 
# import tensorflow as tf
# from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda, MaxPooling2D
# from tensorflow.keras.models import Sequential, Model
import tensorflow.keras.backend as K
from keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda, MaxPooling2D, Reshape
from keras.layers import LSTM, CuDNNLSTM, GRU, Bidirectional, TimeDistributed, Input
from keras.models import Model

def cnnlstmctc(input_shape : Tuple[int, ...],
               output_shape : Tuple[int, ...],
               window_width: float = 28,
               window_stride: float = 7,
               backbone : str = 'lenet',
               seq_model : str = 'lstm',
               bi : bool = False) -> Model:
    """
    Creates a cnn lstm model 
    
    Args:
    input_shape : shape of the input tensor
    output_shape : shape of the output tensor
    window_width : width of sliding window
    window_stride : stride of sliding window
    network : backbone network
    seq_model : whether to use LSTM or GRU
    bi : whether to use bidirectional wrapper
    
    Returns:
    CNN LSTM Model
    """

    image_height, image_width = input_shape    # 28, 952
    output_length, num_classes = output_shape  # 34, 64
    num_windows = int((image_width - window_width) / window_stride) + 1 #118

    if num_windows < output_length:
        raise ValueError(f'Window width/stride need to generate >= {output_length} windows (currently {num_windows})')

    # Input (28, 952) -> Output (28, 952, 1)
    image_input = Input(name='inputs', shape=input_shape, dtype='float32')
    labels = Input(name='labels', shape=(output_length,), dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    
    image_reshaped = Reshape((image_height, image_width, 1))(image_input)
    image_patches = Lambda(
        slide_window,
        arguments={'window_width': window_width, 'window_stride': window_stride}
    )(image_reshaped)
    # (118, 28, 16, 1)
    
    func = {"lstm" : LSTM, "gru" : GRU, "lenet" : lenet, "resnet" : resnet, "custom" : customCNN}

    gpu_present = len(device_lib.list_local_devices()) > 2
    lstm_fn = CuDNNLSTM if gpu_present and seq_model == "lstm" else func[seq_model]
    lstm_fn = CuDNNGRU if gpu_present and seq_model == "gru" else func[seq_model]

    network_fn = func[backbone]

    # Any backbone model without the last two layers (softmax and dropout)
    convnet = network_fn((image_height, window_width, 1), (num_classes,))
    convnet = Model(inputs=convnet.inputs, outputs=convnet.layers[-2].output)
    convnet_outputs = TimeDistributed(convnet)(image_patches)
    # (118, 128)

    if bi :
        lstm_output = Bidirectional(lstm_fn(128, return_sequences=True))(convnet_outputs)
    else:
        lstm_output = lstm_fn(128, return_sequences=True)(convnet_outputs)
    # (118, 128)

    softmax_output = Dense(num_classes, activation='softmax', name='softmax_output')(lstm_output)
    # (118, 64)

    input_length_processed = Lambda(
        lambda x, num_windows=None: x * num_windows,
        arguments={'num_windows': num_windows}
    )(input_length)

    ctc_loss_output = Lambda(
        lambda x: K.ctc_batch_cost(x[0], x[1], x[2], x[3]),
        name='ctc_loss'
    )([labels, softmax_output, input_length_processed, label_length])

    ctc_decoded_output = Lambda(
        lambda x: ctc_decode(x[0], x[1], output_length),
        name='ctc_decoded'
    )([softmax_output, input_length_processed])

    model = Model(
        inputs=[image_input, labels, input_length, label_length],
        outputs=[ctc_loss_output, ctc_decoded_output]
    )

    return model    