"""
IAM Lines dataset. Downloads IAM dataset and saves as .h5 file if not already present.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from urllib.request import urlretrieve
import zipfile
from scipy.io import loadmat
import os
import shutil
import h5py
import  errno
import numpy as np
import json
from pathlib import Path
from tensorflow.keras.utils import to_categorical
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.data.dataset import Dataset
from src.data.emnist_dataset import EMNIST
from src import util

root_folder = Path(__file__).resolve().parents[2]/'data'
raw_folder = root_folder/'raw'
processed_folder = root_folder/'processed'
url = 'https://s3-us-west-2.amazonaws.com/fsdl-public-assets/iam_lines.h5'
filename = root_folder/'processed'/'iam_lines.h5'


class IAMLines(Dataset):
    """
    "The IAM Lines dataset, first published at the ICDAR 1999, contains forms of unconstrained handwritten text,
    which were scanned at a resolution of 300dpi and saved as PNG images with 256 gray levels.
    From http://www.fki.inf.unibe.ch/databases/iam-handwriting-database
    The data split we will use is
    IAM lines Large Writer Independent Text Line Recognition Task (lwitlrt): 9,862 text lines.
        The validation set has been merged into the train set.
        The train set has 7,101 lines from 326 writers.
        The test set has 1,861 lines from 128 writers.
        The text lines of all data sets are mutually exclusive, thus each writer has contributed to one set only.
    """
    def __init__(self):
        
        self.dataset = EMNIST()
        (_, _), (_, _) = self.dataset.load_data()
        self.mapping = augment_emnist_mapping(self.dataset.mapping)
        self.inverse_mapping = {v: k for k, v in self.mapping.items()}
        self.input_shape = (self.dataset.input_shape[0], self.dataset.input_shape[1] * 34)
        self.num_classes = len(self.mapping)
        self.output_shape = (97, self.num_classes)

        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def download(self):
        """Download IAM Lines dataset"""

        try:
            os.makedirs(raw_folder)
            os.makedirs(processed_folder)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        print('[INFO] Downloading raw dataset...')
        util.download_url(url, filename)
        print ('[INFO] Download complete..')

    def load_data(self):
        """ Load IAM Lines dataset"""

        PROCESSED_DATA_DIRNAME = processed_folder
        PROCESSED_DATA_FILENAME = PROCESSED_DATA_DIRNAME/'iam_lines.h5'

        if not os.path.exists(PROCESSED_DATA_FILENAME):
            self.download()
        with h5py.File(PROCESSED_DATA_FILENAME, 'r') as f:
            self.x_train = f['x_train'][:]
            self.y_train = f['y_train'][:]
            self.x_test = f['x_test'][:]
            self.y_test = f['y_test'][:]
        
        self.y_train = to_categorical(self.y_train, num_classes=self.num_classes)
        self.y_test = to_categorical(self.y_test, num_classes=self.num_classes)
        
        return (self.x_train, self.y_train), (self.x_test, self.y_test)

    def __repr__(self):
        return (
            'IAM Lines Dataset\n'
            f'Num classes: {self.num_classes}\n'
            f'Mapping: {self.mapping}\n'
            f'Input shape: {self.input_shape}\n'
        )  

def augment_emnist_mapping(mapping):
    """Augment the mapping with extra symbols."""
    # Extra symbols in IAM dataset
    extra_symbols = [' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '?']

    # padding symbol
    extra_symbols.append('_')

    max_key = max(mapping.keys())
    extra_mapping = {}
    for i, symbol in enumerate(extra_symbols):
        extra_mapping[max_key + 1 + i] = symbol

    return {**mapping, **extra_mapping}

def main():
    """Load IAM Lines dataset and print INFO."""

    dataset = IAMLines()
    (x_train, y_train), (x_test, y_test) = dataset.load_data()

    print(dataset)
    print('Training shape:', x_train.shape, y_train.shape)
    print('Test shape:', x_test.shape, y_test.shape)

if __name__ == '__main__':
    main()