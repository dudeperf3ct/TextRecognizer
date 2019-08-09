from pathlib import Path
import shutil

import numpy as np
import imageio
import sys
sys.path.append(str(Path(__file__).resolve().parents[3]))
from src.data.emnist_dataset import EMNIST

SUPPORT_DIRNAME = Path(__file__).parents[0].resolve() / 'emnist'


def create_emnist_support_files():
    shutil.rmtree(SUPPORT_DIRNAME, ignore_errors=True)
    SUPPORT_DIRNAME.mkdir()

    dataset = EMNIST()
    (_, _), (_, _) = dataset.load_data()

    for ind in [1, 3, 5, 7, 9]:
        image = dataset.x_test[ind]
        label = dataset.mapping[np.argmax(dataset.y_test[ind])]
        print(ind, label)
        imageio.imwrite(str(SUPPORT_DIRNAME / f'{label}.png'), image)

if __name__ == '__main__':
    create_emnist_support_files()