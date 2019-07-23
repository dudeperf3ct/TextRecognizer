"""
EMNIST dataset. Downloads from NIST website and saves as .npz file if not already present.
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
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.data.dataset import Dataset
from src import util

root_folder = Path().absolute().parents[1]/'data'
raw_folder = root_folder/'raw'
processed_folder = root_folder/'processed'
url = 'https://s3-us-west-2.amazonaws.com/fsdl-public-assets/matlab.zip'
filename = raw_folder/'matlab.zip'
ESSENTIALS_FILENAME = raw_folder/'emnist_essentials.json'

class EMNIST(Dataset):
    """
    "The EMNIST dataset is a set of handwritten character digits derived from the NIST Special Database 19
    and converted to a 28x28 pixel image format and dataset structure that directly matches the MNIST dataset."
    From https://www.nist.gov/itl/iad/image-group/emnist-dataset
    The data split we will use is
    EMNIST ByClass: 814,255 characters. 62 unbalanced classes.
    """
    def __init__(self):
        
        if os.path.exists(ESSENTIALS_FILENAME):
            with open(ESSENTIALS_FILENAME) as f:
                essentials = json.load(f)
            self.mapping = dict(essentials['mapping'])
            self.num_classes = len(self.mapping)
            self.input_shape = essentials['input_shape']
            self.output_shape = (self.num_classes,)
        
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def download(self):
        """Download EMNIST dataset"""

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
        
        print('[INFO] Unzipping raw dataset...')
        zip_file = zipfile.ZipFile(filename, 'r')
        zip_file.extract('matlab/emnist-byclass.mat', processed_folder)
        print ('[INFO] Unzipping complete')

        print('[INFO] Loading training and test data from .mat file...')
        data = loadmat(processed_folder/'matlab/emnist-byclass.mat')
        x_train = data['dataset']['train'][0, 0]['images'][0, 0].reshape(-1, 28, 28).swapaxes(1, 2)
        y_train = data['dataset']['train'][0, 0]['labels'][0, 0]
        x_test = data['dataset']['test'][0, 0]['images'][0, 0].reshape(-1, 28, 28).swapaxes(1, 2)
        y_test = data['dataset']['test'][0, 0]['labels'][0, 0]
        
        print('[INFO] Saving to HDF5 in a compressed format...')
        PROCESSED_DATA_DIRNAME = processed_folder
        PROCESSED_DATA_FILENAME = PROCESSED_DATA_DIRNAME/'byclass.h5'

        with h5py.File(PROCESSED_DATA_FILENAME, 'w') as f:
            f.create_dataset('x_train', data=x_train, dtype='u1', compression='lzf')
            f.create_dataset('y_train', data=y_train, dtype='u1', compression='lzf')
            f.create_dataset('x_test', data=x_test, dtype='u1', compression='lzf')
            f.create_dataset('y_test', data=y_test, dtype='u1', compression='lzf')       
    
        print('[INFO] Saving essential dataset parameters...')
        mapping = {int(k): chr(v) for k, v in data['dataset']['mapping'][0, 0]}
        essentials = {'mapping': list(mapping.items()), 'input_shape': list(x_train.shape[1:])}
        self.mapping = mapping    
        self.num_classes = len(self.mapping)
        self.input_shape = essentials['input_shape']
        self.output_shape = (self.num_classes,)

        with open(ESSENTIALS_FILENAME, 'w') as f:
            json.dump(essentials, f)

        print('[INFO] Cleaning up...')
        os.remove(filename)
        shutil.rmtree(processed_folder/'matlab')

    def load_data(self):
        """ Load EMNIST dataset"""

        PROCESSED_DATA_DIRNAME = processed_folder
        PROCESSED_DATA_FILENAME = PROCESSED_DATA_DIRNAME/'byclass.h5'

        if not os.path.exists(PROCESSED_DATA_FILENAME):
            self.download()
        with h5py.File(PROCESSED_DATA_FILENAME, 'r') as f:
            self.x_train = f['x_train'][:]
            self.y_train = f['y_train'][:]
            self.x_test = f['x_test'][:]
            self.y_test = f['y_test'][:]

        return (x_train, y_train), (x_test, y_test)

    def __repr__(self):
        return (
            'EMNIST Dataset\n'
            f'Num classes: {self.num_classes}\n'
            f'Mapping: {self.mapping}\n'
            f'Input shape: {self.input_shape}\n'
        )  


def main():
    """Load EMNIST dataset and print INFO."""

    dataset = EMNIST()
    (x_train, y_train), (x_test, y_test)= dataset.load_data()

    print(dataset)
    print('Training shape:', x_train.shape, y_train.shape)
    print('Test shape:', x_test.shape, y_test.shape)

if __name__ == '__main__':
    main()