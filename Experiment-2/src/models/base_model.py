"""
Base Model class
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import errno
import numpy as np
np.random.seed(42)

from pathlib import Path
from typing import Callable, Dict, Optional, Tuple
from keras.optimizers import RMSprop, Adam
#from tensorflow.keras.optimizers import RMSprop, Adam

WEIGHTS_DIR = Path(__file__).parents[2].resolve() / 'models'
MODEL_DIR = Path(__file__).parents[2].resolve() / 'models'


class Model:
    """
    Base class
    """
    def __init__(self,
                 network_fn : Callable,
                 dataset : type,
                 network_args : Dict = None):
        """
        Args:
        Network_fn -> Type of network class
        dataset -> Type of dataset class
        network_args -> Arguments to pass to network
        """
        self.name = f'{self.__class__.__name__}_{dataset.__name__}_{network_fn.__name__}'
        self.data = dataset()
        if network_args is None:
            network_args = {}
        self.network = network_fn(self.data.input_shape, self.data.output_shape, **network_args)

    @property
    def image_shape(self) -> Tuple[int, ...]:
        return self.data.input_shape

    @property
    def weights_filename(self) -> str:
        try:
            os.makedirs(WEIGHTS_DIR)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
        return str(WEIGHTS_DIR/f'{self.name}_weights.h5')

    @property
    def model_filename(self) -> str:
        try:
            os.makedirs(MODEL_DIR)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
        return str(MODEL_DIR/f'{self.name}_model.h5')
    
    def train_generator(self, dataset, shuff_index, batch_size):
        num_iters = int(np.ceil(dataset['x_train'].shape[0] / batch_size))
        while 1:
            for i in range(num_iters):
                idx = shuff_index[i*batch_size:(i+1)*batch_size]
                tmp = dataset['x_train'][idx].astype('float32')
                tmp -= np.mean(dataset['x_train'], axis=0, keepdims=True)
                tmp /= 255.0
                if network_args is not None:
                    x, y = format_batch_ctc(tmp, dataset['y_train'][idx])
                else:
                    x, y = tmp, dataset['y_train'][idx]
                yield x, y                 
    
    def valid_generator(self, dataset, batch_size):
        num_iters = int(np.ceil(dataset['x_valid'].shape[0] / batch_size))
        while 1:
            for i in range(num_iters):
                tmp = dataset['x_valid'][i*batch_size:(i+1)*batch_size].astype('float32')
                tmp -= np.mean(dataset['x_train'], axis=0, keepdims=True)
                tmp /= 255.0
                if network_args is not None:
                    x, y = format_batch_ctc(tmp, dataset['y_valid'][i*batch_size:(i+1)*batch_size])
                else:
                    x, y = tmp, dataset['y_valid'][i*batch_size:(i+1)*batch_size]
                yield x, y 

    def fit(self, dataset, batch_size : int = 32, epochs : int = 10, callbacks : list = None, lr : float = 1e-3):
        if callbacks is None:
            callbacks = []
        #compile the network
        self.network.compile(loss=self.loss(), optimizer=self.optimizer(lr), metrics=self.metrics())
        #get the batches from generator
        shuff_index = np.random.permutation(dataset['x_train'].shape[0])
        trn_generator = self.train_generator(dataset, shuff_index, batch_size=batch_size)
        val_generator = self.valid_generator(dataset, batch_size=batch_size)

        iters_train = int(np.ceil(dataset['x_train'].shape[0] / float(batch_size)))
        iters_test = int(np.ceil(dataset['x_valid'].shape[0] / float(batch_size)))
        print ('Number:', iters_train, iters_test)
        #train the model using fit_generator
        history = self.network.fit_generator(
                    generator=trn_generator,
                    steps_per_epoch=iters_train,
                    epochs=epochs,
                    callbacks=callbacks,
                    validation_data=val_generator,
                    validation_steps=iters_test,
                    use_multiprocessing=True,
                    verbose=2
                )
        return history

    def test_generator(self, dataset, batch_size : int):
        num_iters = int(np.ceil(dataset['x_test'].shape[0] / batch_size))
        while 1:
            for i in range(num_iters):
                tmp = dataset['x_test'][i*batch_size:(i+1)*batch_size].astype('float32')
                tmp /= 255.0
                yield tmp, dataset['y_test'][i*batch_size:(i+1)*batch_size]

    def evaluate(self, dataset, batch_size : int = 16):
        t_generator = self.test_generator(dataset, batch_size=batch_size)
        iters_test = int(np.ceil(dataset['x_test'].shape[0] / float(batch_size)))
        loss, accuracy = self.network.evaluate_generator(t_generator, steps=iters_test, verbose=2)
        return loss, accuracy

    def loss(self):
        return 'categorical_crossentropy'

    def optimizer(self, lr):
        return Adam(lr, amsgrad=True)

    def metrics(self):
        return ['accuracy']

    def load_weights(self):
        self.network.load_weights(self.weights_filename)

    def save_weights(self):
        self.network.save_weights(self.weights_filename)
        print ('[INFO] Weights saved at', self.weights_filename)

    def save_model(self):
        self.network.save(self.model_filename)
        print ('[INFO] Model saved at', self.model_filename) 
        
        
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
