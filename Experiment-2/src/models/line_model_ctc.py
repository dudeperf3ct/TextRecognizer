"""
Character Model class
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import editdistance
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple
from src.models.base_model import Model
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model as KerasModel
from src.data.emnist_lines import EMNISTLines
from src.networks.cnn_lstm_ctc import cnnlstmctc

class LineModelCTC(Model):
    """
    Character Model class
    """
    def __init__(self,
                 network_fn : Callable = cnnlstmctc,
                 dataset : type = EMNISTLines,
                 network_args : Dict = None):
        """Model for recognizing handwritten text in an image of a line, using CTC loss/decoding."""

        default_network_args = {'backbone': 'lenet', 'seq_model': 'lstm', 'bi':False}
        if network_args is None:
            network_args = {}
        network_args = {**default_network_args, **network_args}
        super().__init__(network_fn, dataset, network_args) 

    def evaluate(self, dataset, batch_size=16, verbose=True):
        
        iters_test = int(np.ceil(dataset['x_test'].shape[0] / float(batch_size)))
        test_gen = test_generator(dataset, batch_size)

        # We can use the `ctc_decoded` layer that is part of our model here.
        decoding_model = KerasModel(inputs=self.network.input, outputs=self.network.get_layer('ctc_decoded').output)
        preds = decoding_model.predict_generator(test_gen, steps=iters_test, verbose=2)
        
        trues = np.argmax(dataset['y_test'], -1)

        pred_strings = [''.join(self.data.mapping.get(label, '') for label in pred).strip(' |_') for pred in preds]
        true_strings = [''.join(self.data.mapping.get(label, '') for label in true).strip(' |_') for true in trues]
        
        char_accuracies = [
            1 - editdistance.eval(true_string, pred_string) / len(true_string)
            for pred_string, true_string in zip(pred_strings, true_strings)
        ]
        mean_accuracy = np.mean(char_accuracies)
        
        if verbose:
            sorted_ind = np.argsort(char_accuracies)
            print("\nLeast accurate predictions:")
            for ind in sorted_ind[:5]:
                print(f'True: {true_strings[ind]}')
                print(f'Pred: {pred_strings[ind]}')
            print("\nMost accurate predictions:")
            for ind in sorted_ind[-5:]:
                print(f'True: {true_strings[ind]}')
                print(f'Pred: {pred_strings[ind]}')
            print("\nRandom predictions:")
            random_ind = np.random.randint(0, len(char_accuracies), 5)
            for ind in random_ind:
                print(f'True: {true_strings[ind]}')
                print(f'Pred: {pred_strings[ind]}')

        return mean_accuracy        
    
    
    def predict_on_image(self, image: np.ndarray) -> Tuple[str, float]:
        softmax_output_fn = K.function(
            [self.network.get_layer('inputs').input, K.learning_phase()],
            [self.network.get_layer('softmax_output').output]
        )
        if image.dtype == np.uint8:
            image = (image / 255).astype(np.float32)

        # Get the prediction and confidence using softmax_output_fn, passing the right input into it.
        input_image = np.expand_dims(image, 0)
        softmax_output = softmax_output_fn([input_image, 0])[0]

        input_length = np.array([softmax_output.shape[1]])
        decoded, log_prob = K.ctc_decode(softmax_output, input_length, greedy=True)

        pred_raw = K.eval(decoded[0])[0]
        pred = ''.join(self.data.mapping[label] for label in pred_raw).strip()

        neg_sum_logit = K.eval(log_prob)[0][0]
        conf = np.exp(-neg_sum_logit)

        return pred, conf   
        
    def loss(self):
        """Dummy loss function: just pass through the loss that we computed in the network."""
        return {'ctc_loss': lambda y_true, y_pred: y_pred}

    def metrics(self):
        """We could probably pass in a custom character accuracy metric for 'ctc_decoded' output here."""
        return None

def test_generator(self, dataset, batch_size : int):
    num_iters = int(np.ceil(dataset['x_test'].shape[0] / batch_size))
    while 1:
        for i in range(num_iters):
            tmp = dataset['x_test'][i*batch_size:(i+1)*batch_size].astype('float32')
            tmp /= 255.0
            x, y = format_batch_ctc(tmp, dataset['y_test'][i*batch_size:(i+1)*batch_size])
            yield x, y


def format_batch_ctc(batch_x, batch_y):
    """
    Because CTC loss needs to be computed inside of the network, we include information about outputs in the inputs.
    """
    batch_size = batch_y.shape[0]
    y_true = np.argmax(batch_y, axis=-1)

    label_lengths = []
    for ind in range(batch_size):
        # Find all of the indices in the label that are blank
        empty_at = np.where(batch_y[ind, :, -1] == 1)[0]
        # Length of the label is the pos of the first blank, or the max length
        if empty_at.shape[0] > 0:
            label_lengths.append(empty_at[0])
        else:
            label_lengths.append(batch_y.shape[1])

    batch_inputs = {
        'image': batch_x,
        'y_true': y_true,
        'input_length': np.ones((batch_size, 1)),  # dummy, will be set to num_windows in network
        'label_length': np.array(label_lengths)
    }
    batch_outputs = {
        'ctc_loss': np.zeros(batch_size),  # dummy
        'ctc_decoded': y_true
    }

    return batch_inputs, batch_outputs