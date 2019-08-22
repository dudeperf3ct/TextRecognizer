"""
IAM dataset. Downloads IAM dataset.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Dict, List
from urllib.request import urlretrieve
from boltons.cacheutils import cachedproperty
import zipfile
import os
import xml.etree.ElementTree as ElementTree
import  errno
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.data.dataset import Dataset
from src import util

root_folder = Path(__file__).resolve().parents[2]/'data'
raw_folder = root_folder/'raw'
url = 'https://s3-us-west-2.amazonaws.com/fsdl-public-assets/iam/iamdb.zip'
RAW_DATASET_DIRNAME = raw_folder/'iamdb'
filename = raw_folder/'iamdb.zip'

DOWNSAMPLE_FACTOR = 2  # If images were downsampled, the regions must also be.
LINE_REGION_PADDING = 0  # add this many pixels around the exact coordinates

class IAMDataset(Dataset):
    """
    The IAM Lines dataset, first published at the ICDAR 1999, contains forms of unconstrained handwritten text,
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
        pass

    def download(self):
        """Download IAM Lines dataset"""
        try:
            os.makedirs(raw_folder)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        print('[INFO] Downloading raw dataset...')
        util.download_url(url, filename)
        print ('[INFO] Download complete..')

        print('[INFO] Extracting IAM data...')
        with zipfile.ZipFile(filename, 'r') as zip_file:
            zip_file.extractall(raw_folder)
        print ('[INFO] Extraction complete..')

    @property
    def xml_filenames(self):
        return list((RAW_DATASET_DIRNAME / 'xml').glob('*.xml'))

    @property
    def form_filenames(self):
        return list((RAW_DATASET_DIRNAME / 'forms').glob('*.jpg'))

    @property
    def form_filenames_by_id(self):
        return {filename.stem: filename for filename in self.form_filenames}

    @cachedproperty
    def line_strings_by_id(self):
        """Return a dict from name of IAM form to a list of line texts in it."""
        return {
            filename.stem: get_line_strings_from_xml_file(filename)
            for filename in self.xml_filenames
        }

    @cachedproperty
    def line_regions_by_id(self):
        """Return a dict from name of IAM form to a list of (x1, x2, y1, y2) coordinates of all lines in it."""
        return {
            filename.stem: get_line_regions_from_xml_file(filename)
            for filename in self.xml_filenames
        }

    def load_data(self):
        """ Load IAM dataset"""
        if not self.xml_filenames:
            self.download()

    def __repr__(self):
        return (
            'IAM Dataset\n'
            f'Num forms: {len(self.xml_filenames)}\n'
        )  

def get_line_strings_from_xml_file(filename: str) -> List[str]:
    """Get the text content of each line. Note that we replace &quot; with "."""
    xml_root_element = ElementTree.parse(filename).getroot()  # nosec
    xml_line_elements = xml_root_element.findall('handwritten-part/line')
    return [el.attrib['text'].replace('&quot;', '"') for el in xml_line_elements]


def get_line_regions_from_xml_file(filename: str) -> List[Dict[str, int]]:
    """Get the line region dict for each line."""
    xml_root_element = ElementTree.parse(filename).getroot()  # nosec
    xml_line_elements = xml_root_element.findall('handwritten-part/line')
    return [get_line_region_from_xml_element(el) for el in xml_line_elements]


def get_line_region_from_xml_element(xml_line) -> Dict[str, int]:
    """
    line (xml element): has x, y, width, and height attributes
    """
    word_elements = xml_line.findall('word/cmp')
    x1s = [int(el.attrib['x']) for el in word_elements]
    y1s = [int(el.attrib['y']) for el in word_elements]
    x2s = [int(el.attrib['x']) + int(el.attrib['width']) for el in word_elements]
    y2s = [int(el.attrib['y']) + int(el.attrib['height']) for el in word_elements]
    return {
        'x1': min(x1s) // DOWNSAMPLE_FACTOR - LINE_REGION_PADDING,
        'y1': min(y1s) // DOWNSAMPLE_FACTOR - LINE_REGION_PADDING,
        'x2': max(x2s) // DOWNSAMPLE_FACTOR + LINE_REGION_PADDING,
        'y2': max(y2s) // DOWNSAMPLE_FACTOR + LINE_REGION_PADDING
    }


def main():
    """Load IAM Dataset dataset and print INFO."""

    dataset = IAMDataset()
    dataset.load_data()

    print(dataset)

if __name__ == '__main__':
    main()