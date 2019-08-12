"""CharacterPredictor class"""
from typing import Tuple, Union

import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.character_model import Character_Model
from src.networks.lenet import lenet
from src.networks.resnet import resnet
from src.networks.custom import customCNN
import imageio
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class CharacterPredictor:
    """Given an image of a single handwritten character, recognizes it."""
    def __init__(self):
        self.model = Character_Model(customCNN)
        self.model.load_weights()

    def predict(self, image_or_filename: Union[np.ndarray, str]) -> Tuple[str, float]:
        """Predict on a single image."""
        if isinstance(image_or_filename, str):
            image = imageio.imread(image_or_filename, pilmode='L')
        else:
            image = image_or_filename
        return self.model.predict_on_image(image)

    def evaluate(self, dataset):
        """Evaluate on a dataset."""
        return self.model.evaluate(dataset.x_test, dataset.y_test)