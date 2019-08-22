"""LinePredictor class"""
from typing import Tuple, Union

import numpy as np
from src.models.line_model_ctc import LineModelCTC
from src.data.iam_lines import IAMLines
import imageio
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class LinePredictor:
    """Given an image of a line of handwritten text, recognizes text contents."""
    def __init__(self, dataset_cls=IAMLines):
        self.model = LineModelCTC(dataset_cls=dataset_cls)
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
