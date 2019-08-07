"""Function to train a model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
#from tensorflow.keras.callbacks import EarlyStopping
from keras.callbacks import EarlyStopping
from src.data.dataset import Dataset
from src.models.base_model import Model
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
#from src.visualization.visualize import plot_history, save_model
from src.clr_callback import CyclicLR

EARLY_STOPPING = True
CYCLIC_LR = True
MIN_LR = 1e-7
MAX_LR = 1e-2
STEP_SIZE = 8
MODE = "triangular"

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

    if CYCLIC_LR:
        cyclic_lr = CyclicLR(base_lr=MIN_LR, max_lr=MAX_LR,
                             step_size=STEP_SIZE * (dataset['x_train'].shape[0] // batch_size), 
                             mode=MODE)
        callbacks.append(cyclic_lr)

    model.network.summary()

    t = time.time()
    _history = model.fit(dataset=dataset, 
                         batch_size=batch_size, 
                         epochs=epochs, 
                         callbacks=callbacks)
    print('[INFO] Training took {:2f} s'.format(time.time() - t))

    #plot_history(_history)
    #save_model(model.network)

    return model