"""LinePredictor class"""
from typing import Tuple, Union

import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.models.line_model_ctc import LineModelCTC
from src.data.iam_lines import IAMLines
import imageio
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class LinePredictor:
    """Given an image of a line of handwritten text, recognizes text contents."""
    def __init__(self, dataset=IAMLines):
        args = {'backbone' : 'lenet', 'seq_model' : 'lstm', 'bi' : True}
        self.model = LineModelCTC(dataset=IAMLines, network_args=args)
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
