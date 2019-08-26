"""
Line Detector Model class
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Callable, Dict, Tuple
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.data.iam_paragraphs import IAMPara
from src.models.base_model import Model
from src.networks.lenet_fcn import lenetFCN
from src.networks.fcn import FCN

DATA_AUGMENTATION_PARAMS = {
    'width_shift_range': 0.06,
    'height_shift_range': 0.1,
    'horizontal_flip': True,
    'zoom_range': 0.1,
    'fill_mode': 'constant',
    'cval': 0,
    'shear_range': 3,
}


class LineDetectModel(Model):
    """Model to detect lines of text in an image."""
    def __init__(self,
                 network_fn: Callable = FCN,
                 dataset: type = IAMPara):

        """Define the default dataset and network values for this model."""
        super().__init__(network_fn, dataset)
        print ('[INFO] Arguments passed to data augmentation...', DATA_AUGMENTATION_PARAMS)
        self.data_augmentor = ImageDataGenerator(**DATA_AUGMENTATION_PARAMS)
        self.batch_augment_fn = self.augment_batch

    def metrics(self):
        return None

    def augment_batch(self, x_batch: np.ndarray, y_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Performs different random transformations on the whole batch of x, y samples."""
        x_augment, y_augment = zip(*[self.augment_sample(x, y) for x, y in zip(x_batch, y_batch)])
        return np.stack(x_augment, axis=0), np.stack(y_augment, axis=0)

    def augment_sample(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform the same random image transformation on both x and y.
        x is a 2d image of shape self.image_shape, but self.data_augmentor needs the channel image too.
        """
        x_3d = np.expand_dims(x, axis=-1)
        transform_parameters = self.data_augmentor.get_random_transform(x_3d.shape)
        x_augment = self.data_augmentor.apply_transform(x_3d, transform_parameters)
        y_augment = self.data_augmentor.apply_transform(y, transform_parameters)
        return np.squeeze(x_augment, axis=-1), y_augment

    def predict_on_image(self, x: np.ndarray) -> np.ndarray:
        """Returns the network predictions on x."""
        return self.network.predict(np.expand_dims(x, axis=0))[0]