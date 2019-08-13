"""Implements Sliding window"""

import tensorflow as tf

def slide_window(image, window_width, window_stride):
    """
    Takes input of shape    -> (image_height, image_width, 1) ,
    Returns output of shape -> (num_windows, image_height, window_width, 1)
    num_windows ->  floor((image_width - window_width) / window_stride) + 1
    """
    kernel = [1, 1, window_width, 1]
    strides = [1, 1, window_stride, 1]
    patches = tf.extract_image_patches(image, kernel, strides, [1, 1, 1, 1], 'VALID')
    patches = tf.transpose(patches, (0, 2, 1, 3))
    patches = tf.expand_dims(patches, -1)
    return patches