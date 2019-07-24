"""Function to train a model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from tensorflow.keras.callbacks import EarlyStopping
from src.datasets.dataset import Dataset
from src.models.base import Model
#from src.visualization.visualize import plot_history

EARLY_STOPPING = True

def train_model(
        model: Model,
        dataset: Dataset,
        epochs: int,
        batch_size: int) -> Model:
    """Train model."""
    callbacks = []

    if EARLY_STOPPING:
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, 
                         patience=3, verbose=1, mode='auto')
        callbacks.append(early_stopping)

    model.network.summary()

    t = time()
    _history = model.fit(dataset=dataset, batch_size=batch_size, epochs=epochs, callbacks=callbacks)
    print('[INFO] Training took {:2f} s'.format(time() - t))

    #plot_history(_history)

    return model