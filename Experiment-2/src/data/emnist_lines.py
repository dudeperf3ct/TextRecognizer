"""
EMNIST Lines dataset. Uses EMNIST dataset to create EMNIST Lines dataset.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import h5py
from pathlib import Path
from tensorflow.keras.utils import to_categorical
from collections import defaultdict
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.data.dataset import Dataset
from src.data.emnist_dataset import EMNIST
from src.data.sentence_generator import SentenceGenerator
import numpy as np
np.random.seed(42)

root_folder = Path(__file__).resolve().parents[2]/'data'
raw_folder = root_folder/'raw'
processed_folder = root_folder/'processed'

class EMNISTLines(Dataset):
    """
    Here we will create a synthetic dataset using EMNIST dataset and
    we will use text from Brown Corpus avaliable on NLTK 
    
    Args :
    max_length : Maximum length of the labels
    max_overlap : Overlap to stitch images together
    num_train : Number of train samples
    num_test : Number of test samples
    """
    def __init__(self, max_length : int = 34, max_overlap : float = 0.33, num_train : int = 2, num_test : int = 1):
        self.dataset = EMNIST()
        (_, _), (_, _) = self.dataset.load_data()
        self.mapping = augment_emnist_mapping(self.dataset.mapping)
        self.inverse_mapping = {v: k for k, v in self.mapping.items()}
        self.max_length = max_length
        self.max_overlap = max_overlap
        self.input_shape = (self.dataset.input_shape[0], self.dataset.input_shape[1] * max_length)
        self.num_classes = len(self.mapping)
        self.output_shape = (self.max_length, self.num_classes)
        
        self.num_train = num_train
        self.num_test = num_test
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    @property
    def data_filename(self):
        return processed_folder / f'ml_{self.max_length}_mo{self.max_overlap}_ntr{self.num_train}_nte{self.num_test}.h5'

    def generate_data(self, split : str):

        print('[INFO] Creating EmnistLines Dataset...')
        data = self.dataset
        train_labels = get_labels(data.y_train, data.mapping)
        test_labels = get_labels(data.y_test, data.mapping)
        
        trn_dict = create_data_dict(data.x_train, train_labels, self.mapping)
        test_dict = create_data_dict(data.x_test, test_labels, self.mapping)

        num = self.num_train if split == 'train' else self.num_test
        data_dict = trn_dict if split == 'train' else test_dict

        sent = SentenceGenerator(self.max_length)

        print('[INFO] Saving EmnistLines Dataset to HDF5...')
        with h5py.File(self.data_filename, 'a') as f:
            x, y = create_image_string_dataset(num, data_dict, sent, self.max_overlap)
            y = convert_strings_to_categorical_labels(y, self.inverse_mapping)
            f.create_dataset(f'x_{split}', data=x, dtype='u1', compression='lzf')
            f.create_dataset(f'y_{split}', data=y, dtype='u1', compression='lzf')

    def load_data(self):
        """ Load EMNIST Lines dataset"""
        if not os.path.exists(self.data_filename):
            self.generate_data(split='train')
            self.generate_data(split='test')
 
        print('[INFO] EmnistLines Dataset loading data from HDF5...')
        with h5py.File(self.data_filename, 'r') as f:
            self.x_train = f['x_train'][:]
            self.y_train = f['y_train'][:]
            self.x_test = f['x_test'][:]
            self.y_test = f['y_test'][:]


    def __repr__(self):
        return (
            'EMNIST Lines Dataset\n'
            f'Max length: {self.max_length}\n'
            f'Max overlap: {self.max_overlap}\n'
            f'Num classes: {self.num_classes}\n'
            f'Input shape: {self.input_shape}\n'
            f'Train: {self.x_train.shape} {self.y_train.shape}\n'
            f'Test: {self.x_test.shape} {self.y_test.shape}\n'
        )

def create_data_dict(samples, labels, mapping) -> dict :
    # create a dict where keys are labels and values are image array
    data_dict = defaultdict(list)
    for sample, label in zip(samples, labels.flatten()):
        data_dict[label].append(sample)
    return data_dict

def stitch_image_string(string : str, data_dict : dict, max_overlap : float) -> np.ndarray :
        # get random sample of image for each corresponding character
        selected_images = []
        zero_image = np.zeros((28, 28), np.uint8)
        for char in string:
            samples = data_dict[char]
            rnd_sample = samples[np.random.choice(len(samples))] if samples else zero_image
            selected_images.append(rnd_sample.reshape(28, 28))
        # stitch the selected images to form a uniform image
        overlap = np.random.rand() * max_overlap
        N = len(selected_images)
        H, W = np.array(selected_images)[0].shape
        #print (N, overlap, H, W)
        next_overlap_width = W - int(overlap * W)
        concatenated_image = np.zeros((H, W * N), np.uint8)
        x = 0
        for image in selected_images:
            concatenated_image[:, x:(x + W)] += image
            x += next_overlap_width
        return np.minimum(255, concatenated_image)

def create_image_string_dataset(num, data_dict, sent, max_overlap):
    rnd_sentence = sent.generate()
    sample_image = stitch_image_string(rnd_sentence, data_dict, 0)  # Note that sample_image has 0 overlap
    images = np.zeros((num, sample_image.shape[0], sample_image.shape[1]), np.uint8)
    labels = []
    for n in range(num):
        label = None
        for _ in range(5):  # Try 5 times to generate before actually erroring
            try:
                label = sent.generate()
                break
            except Exception:
                pass
        images[n] = stitch_image_string(label, data_dict, max_overlap)
        labels.append(label)
    return images, labels


def convert_strings_to_categorical_labels(labels, mapping):
    return np.array([
        to_categorical([mapping[c] for c in label], num_classes=len(mapping))
        for label in labels
    ])    

def get_labels(labels, mapping):    
    return np.array([mapping[np.where(labels[i]==1)[0][0]] for i in range(len(labels))])

def augment_emnist_mapping(mapping):
    """Augment the mapping with extra symbols."""
    # Extra symbol ' ' not present in EMNIST
    extra_symbols = [' ']

    # padding symbol
    extra_symbols.append('_')

    max_key = max(mapping.keys())
    extra_mapping = {}
    for i, symbol in enumerate(extra_symbols):
        extra_mapping[max_key + 1 + i] = symbol

    return {**mapping, **extra_mapping}

def main():
    """Load EMNISTLines dataset and print INFO."""

    dataset = EMNISTLines()
    dataset.load_data()

    print(dataset)

if __name__ == '__main__':
    main()