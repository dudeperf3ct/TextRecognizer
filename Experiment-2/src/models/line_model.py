"""
Character Model class
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import editdistance
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple
from src.models.base_model import Model
from src.data.emnist_lines import EMNISTLines
from src.networks.lenet_cnn import lenetcnn

class LineModel(Model):
    """
    Character Model class
    """
    def __init__(self,
                 network_fn : Callable = lenetcnn,
                 dataset : type = EMNISTLines):
        """Define default network class and dataset class"""
        super().__init__(network_fn, dataset)      

    def evaluate(self, dataset, batch_size=16, verbose=True):
        
        iters_test = int(np.ceil(dataset['x_test'].shape[0] / float(batch_size)))
        test_gen = self.test_generator(dataset, batch_size)
        preds_raw = self.network.predict_generator(test_gen, steps=iters_test, verbose=2)
        trues = np.argmax(dataset['y_test'], -1)
        preds = np.argmax(preds_raw, -1)

        pred_strings = [''.join(self.data.mapping.get(label, '') for label in pred).strip(' |_') for pred in preds]
        true_strings = [''.join(self.data.mapping.get(label, '') for label in true).strip(' |_') for true in trues]
        
        char_accuracies = [
            1 - editdistance.eval(true_string, pred_string) / len(true_string)
            for pred_string, true_string in zip(pred_strings, true_strings)
        ]
        mean_accuracy = np.mean(char_accuracies)
        
        if verbose:
            sorted_ind = np.argsort(char_accuracies)
            print("\nLeast accurate predictions:")
            for ind in sorted_ind[:5]:
                print(f'True: {true_strings[ind]}')
                print(f'Pred: {pred_strings[ind]}')
            print("\nMost accurate predictions:")
            for ind in sorted_ind[-5:]:
                print(f'True: {true_strings[ind]}')
                print(f'Pred: {pred_strings[ind]}')
            print("\nRandom predictions:")
            random_ind = np.random.randint(0, len(char_accuracies), 5)
            for ind in random_ind:
                print(f'True: {true_strings[ind]}')
                print(f'Pred: {pred_strings[ind]}')

        return mean_accuracy        
    
    
    def predict_on_image(self, image: np.ndarray) -> Tuple[str, float]:
        if image.dtype == np.uint8:
            image = (image / 255).astype(np.float32)
        pred_raw = self.network.predict(np.expand_dims(image, 0), batch_size=1).flatten()
        ind = np.argmax(pred_raw)
        confidence_of_prediction = pred_raw[ind]
        # integer to character mapping dictionary is self.data.mapping[integer]
        predicted_character = self.data.mapping[ind]
        return predicted_character, confidence_of_prediction    
        
