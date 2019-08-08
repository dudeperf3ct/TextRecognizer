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
from keras.optimizers import RMSprop
#from tensorflow.keras.optimizers import RMSprop

WEIGHTS_DIR = Path(__file__).parents[2].resolve() / 'models'
MODEL_DIR = Path(__file__).parents[2].resolve() / 'models'


class Model:
    """
    Base class
    """
    def __init__(self,
                 network_fn : Callable,
                 dataset : type):
        """Network_fn -> Type of network class and dataset -> Type of dataset class"""
        self.name = f'{self.__class__.__name__}_{dataset.__name__}_{network_fn.__name__}'
        self.data = dataset()
        self.network = network_fn(self.data.input_shape, self.data.output_shape)

        self.batch_augment_fn: Optional[Callable] = None
        self.batch_format_fn: Optional[Callable] = None

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
                yield tmp, dataset['y_train'][idx]
    
    def valid_generator(self, dataset, batch_size):
        num_iters = int(np.ceil(dataset['x_valid'].shape[0] / batch_size))
        while 1:
            for i in range(num_iters):
                tmp = dataset['x_valid'][i*batch_size:(i+1)*batch_size].astype('float32')
                tmp -= np.mean(dataset['x_train'], axis=0, keepdims=True)
                tmp /= 255.0
                yield tmp, dataset['y_valid'][i*batch_size:(i+1)*batch_size]

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
        return RMSprop(lr)

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
