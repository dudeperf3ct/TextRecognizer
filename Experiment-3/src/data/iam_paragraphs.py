"""
IAM Paragraphs dataset. Downloads IAM dataset and saves as .h5 file if not already present.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.data.dataset import Dataset
from src.data.iam_dataset import IAMDataset
from src import util
import imageio
import numpy as np
np.random.seed(42)

root_folder = Path(__file__).resolve().parents[2]/'data'
raw_folder = root_folder/'raw'
processed_folder = root_folder/'processed'

INTERIM_DATA_DIRNAME = raw_folder / 'iam_paragraphs'
DEBUG_CROPS_DIRNAME = INTERIM_DATA_DIRNAME / 'debug_crops'
PROCESSED_DATA_DIRNAME = root_folder / 'processed' / 'iam_paragraphs'
CROPS_DIRNAME = PROCESSED_DATA_DIRNAME / 'crops'
GT_DIRNAME = PROCESSED_DATA_DIRNAME / 'gt'

PARAGRAPH_BUFFER = 50  # pixels in the IAM form images to leave around the lines
TEST_FRACTION = 0.2

class IAMPara(Dataset):
    """
    """
    def __init__(self):
        
        self.iam_dataset = IAMDataset()
        self.iam_dataset.load_data()

        self.num_classes = 3
        self.input_shape = (256, 256)
        self.output_shape = (256, 256, self.num_classes)

        self.x = None
        self.y = None
        self.ids = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def load_data(self):
        """Load or generate dataset data."""
        num_actual = len(list(CROPS_DIRNAME.glob('*.jpg')))
        num_target = len(self.iam_dataset.line_regions_by_id)
        if num_actual < num_target - 2:  # There are a couple of instances that could not be cropped
            self.process_iam_paragraphs()

        self.x, self.y, self.ids = self.load_iam_paragraphs()
        self.train_ind, self.test_ind = get_random_split(self.x.shape[0])
        ids_train, ids_test = self.ids[self.train_ind], self.ids[self.test_ind]
        self.x_train, self.y_train = self.x[ids_train], self.y[ids_train]
        self.x_test, self.y_test = self.x[ids_test], self.y[ids_test]

    def process_iam_paragraphs(self):
        """
        For each page, crop out the part of it that correspond to the paragraph of text, and make sure all crops are
        self.input_shape. The ground truth data is the same size, with a one-hot vector at each pixel
        corresponding to labels 0=background, 1=odd-numbered line, 2=even-numbered line
        """
        crop_dims = self.decide_on_crop_dims()
        CROPS_DIRNAME.mkdir(parents=True, exist_ok=True)
        DEBUG_CROPS_DIRNAME.mkdir(parents=True, exist_ok=True)
        GT_DIRNAME.mkdir(parents=True, exist_ok=True)
        print(f'[INFO] Cropping paragraphs, generating ground truth, and saving debugging images to {DEBUG_CROPS_DIRNAME}...')
        for filename in self.iam_dataset.form_filenames:
            id_ = filename.stem
            line_region = self.iam_dataset.line_regions_by_id[id_]
            crop_paragraph_image(filename, line_region, crop_dims, self.input_shape)

    def decide_on_crop_dims(self):
        """
        Decide on the dimensions to crop out of the form image.
        Since image width is larger than a comfortable crop around the longest paragraph,
        we will make the crop a square form factor.
        And since the found dimensions 610x610 are pretty close to 512x512,
        we might as well resize crops and make it exactly that, which lets us
        do all kinds of power-of-2 pooling and upsampling should we choose to.
        """
        sample_form_filename = self.iam_dataset.form_filenames[0]
        sample_image = imageio.imread(sample_form_filename, pilmode='L')
        max_crop_width = sample_image.shape[1]
        max_crop_height = get_max_paragraph_crop_height(self.iam_dataset.line_regions_by_id)
        assert max_crop_height <= max_crop_width
        crop_dims = (max_crop_width, max_crop_width)
        print(f'[INFO] Max crop width and height were found to be {max_crop_width}x{max_crop_height}...')
        print(f'[INFO] Setting them to {max_crop_width}x{max_crop_width}...')
        return crop_dims

    def load_iam_paragraphs(self):
        print('[INFO] Loading IAM paragraph crops and ground truth from image files...')
        images = []
        gt_images = []
        ids = []
        for filename in CROPS_DIRNAME.glob('*.jpg'):
            id_ = filename.stem
            image = imageio.imread(filename, pilmode='L')
            image = 1. - image / 255

            gt_filename = GT_DIRNAME / f'{id_}.png'
            gt_image = imageio.imread(gt_filename, pilmode='L')

            images.append(image)
            gt_images.append(gt_image)
            ids.append(id_)
        images = np.array(images).astype(np.float32)
        gt_images = to_categorical(np.array(gt_images), 3).astype(np.uint8)
        return images, gt_images, np.array(ids)

    def __repr__(self):
        """Print info about the dataset."""
        return (
            'IAM Paragraphs Dataset\n'
            f'Num classes: {self.num_classes}\n'
            f'Train: {self.x_train.shape} {self.y_train.shape}\n'
            f'Test: {self.x_test.shape} {self.y_test.shape}\n'
        )

def get_random_split(num_total):
    
    num_train = int((1 - TEST_FRACTION) * num_total)
    ind = np.random.permutation(num_total)
    train_ind, test_ind = ind[:num_train], ind[num_train:]
    return train_ind, test_ind

def get_max_paragraph_crop_height(line_regions_by_id):
    heights = []
    for regions in line_regions_by_id.values():
        min_y1 = min(r['y1'] for r in regions) - PARAGRAPH_BUFFER
        max_y2 = max(r['y2'] for r in regions) + PARAGRAPH_BUFFER
        height = max_y2 - min_y1
        heights.append(height)
    return max(heights)


def crop_paragraph_image(filename, line_regions, crop_dims, final_dims):
    image = imageio.imread(filename, pilmode='L')    

    min_y1 = min(r['y1'] for r in line_regions) - PARAGRAPH_BUFFER
    max_y2 = max(r['y2'] for r in line_regions) + PARAGRAPH_BUFFER
    height = max_y2 - min_y1
    crop_height = crop_dims[0]
    buffer = (crop_height - height) // 2

    # Generate image crop
    image_crop = 255 * np.ones(crop_dims, dtype=np.uint8)
    try:
        image_crop[buffer:buffer + height] = image[min_y1:max_y2]
    except Exception as e:
        print(f'[INFO] Rescued {filename}: {e}...')
        return

    # Generate ground truth
    gt_image = np.zeros_like(image_crop, dtype=np.uint8)
    for ind, region in enumerate(line_regions):
        gt_image[
            (region['y1'] - min_y1 + buffer):(region['y2'] - min_y1 + buffer),
            region['x1']:region['x2']
        ] = ind % 2 + 1

    # Generate image for debugging
    cmap = plt.get_cmap('Set1')
    image_crop_for_debug = np.dstack([image_crop, image_crop, image_crop])
    for ind, region in enumerate(line_regions):
        color = [255 * _ for _ in cmap(ind)[:-1]]
        cv2.rectangle(
            image_crop_for_debug,
            (region['x1'], region['y1'] - min_y1 + buffer),
            (region['x2'], region['y2'] - min_y1 + buffer),
            color,
            3
        )
    image_crop_for_debug = cv2.resize(image_crop_for_debug, final_dims, interpolation=cv2.INTER_AREA)
    util.write_image(image_crop_for_debug, DEBUG_CROPS_DIRNAME / f'{filename.stem}.jpg')

    image_crop = cv2.resize(image_crop, final_dims, interpolation=cv2.INTER_AREA)  # Quality interpolation for input
    util.write_image(image_crop, CROPS_DIRNAME / f'{filename.stem}.jpg')

    gt_image = cv2.resize(gt_image, final_dims, interpolation=cv2.INTER_NEAREST)  # No interpolation for labels
    util.write_image(gt_image, GT_DIRNAME / f'{filename.stem}.png')

def main():
    """Load IAM Paragraphs dataset and print INFO."""

    dataset = IAMPara()
    dataset.load_data()

    print(dataset)

if __name__ == '__main__':
    main()