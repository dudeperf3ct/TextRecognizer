"""
Train model
"""

import argparse

def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save-model", type=int, default=False,
        help="(optional) whether or not model should be saved to disk")
    parser.add_argument("-w", "--weights", type=str,
        help="(optional) path to weights file")
    args = vars(parser.parse_args())

    return args