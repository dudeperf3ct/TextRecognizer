"""
Base Model class
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
np.random.seed(42)

from pathlib import Path
from typing import Callable, Dict, Optional
from src.data.emnist_dataset import EMNIST
from src.networks.lenet import lenet

WEIGHTS_DIR = Path(__file__).parents[2].resolve() / 'models'

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
        self.network.summary()

        self.batch_augment_fn: Optional[Callable] = None
        self.batch_format_fn: Optional[Callable] = None

    @property
    def image_shape(self) -> Tuple[int, ...]:
        return self.data.input_shape

    @property
    def weights_filename(self) -> str:
        try:
            os.makedirs(WEIGHTS_DIR)
        return str(WEIGHTS_DIR/f'{self.name}_weights.h5')
    
    def train_generator(self, dataset, shuff_index, batch_size):
        num_iters = train_x.shape[0] / batch_size
        while 1:
            for i in range(num_iters):
                idx = shuff_index[i*batch_size:(i+1)*batch_size]
                tmp = dataset.x_train[idx].astype('float32')
                tmp -= dataset.x_train_mean
                tmp /= 255.0
                yield tmp, dataset.y_train[idx]
    
    def valid_generator(self, dataset, batch_size):
        num_iters = dataset.x_test.shape[0] / batch_size
        while 1:
            for i in range(num_iters):
                tmp = dataset.x_test[i*batch_size:(i+1)*batch_size].astype('float32')
                tmp -= np.mean(dataset.x_train, axis=0, keepdims=True)
                tmp /= 255.0
                yield tmp, dataset.y_test[i*batch_size:(i+1)*batch_size]

    def fit(self, dataset, batch_size : int = 32, epochs : int = 10, callbacks : list = None):
        if callbacks is None:
            callbacks = []
        #compile the network
        self.network.compile(loss=self.loss(), optimizer=self.optimizer(), metrics=self.metrics())
        #get the batches from generator
        shuff_index = np.random.permutation(dataset.x_train.shape[0])
        trn_generator = train_generator(dataset, shuff_index, batch_size=batch_size)
        val_generator = valid_generator(dataset, batch_size=batch_size)
        #train using fit_generator
        self.network.fit_generator(
            generator=trn_generator,
            steps_per_epoch=int(np.ceil(len(trn_generator)/batch_size)),
            epochs=epochs,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_step=int(np.ceil(len(val_generator)/batch_size)),
            use_multiprocessing=False,
            shuffle=True
        )

    def evaluate(self, dataset, batch_size=16, verbose=False):
        val_x, val_y = valid_generator(dataset, batch_size=batch_size)  # Use a small batch size to use less memory
        preds = self.network.predict_generator(val_x)
        return np.mean(np.argmax(preds, -1) == np.argmax(val_y, -1))

    def loss(self):
        return 'categorical_crossentropy'

    def optimizer(self):
        return RMSprop()

    def metrics(self):
        return ['accuracy']

    def load_weights(self):
        self.network.load_weights(self.weights_filename)

    def save_weights(self):
        self.network.save_weights(self.weights_filename)
