from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
#from tensorflow.keras.utils import plot_model
from keras.utils import plot_model
SAVE_PLOT = '../models/'


def plot_acc(history, name):
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(SAVE_PLOT + 'acc_'+ str(name) + '.png')

def plot_loss(history, name):    
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(SAVE_PLOT + 'loss_'+ str(name) + '.png')

def save_model(model, name):
    plot_model(model, to_file=SAVE_PLOT + str(name)+ '_model.png')