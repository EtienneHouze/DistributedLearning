from __future__ import absolute_import, division, print_function

import csv

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from os.path import join, basename, dirname, isfile


def moving_average(x, size):
    """
    helper function to compute the moving average
    :param x: input array
    :type x: 1D array-like
    :param size: size of the moving averaging window
    :type size: integet
    :return: averaged array
    :rtype: 1D array-like, same size as the input
    """
    window = np.ones(int(max(1, size))) / float(max(1, size))
    return np.convolve(x, window, 'same')


def visualize_csvlog(filepath, **kwargs):
    """
    A function to vizualize  logs as registered in the logs subfolder of models
    :param filepath: the path to the file
    :param metrics (optional): for now, either "loss" or "IoU"
    :param mode (optional): for now, either 'train' or 'val'
    :param 'scale' (optional): either 'linear' or 'log'
    :param smoothing (optional): smoothing factor for the data points
    :return: 
    """
    label_x = 'iterations'
    label_y = ''
    title = ''
    scale = 'linear'
    smoothing = 0
    for key in kwargs.keys():
        if key == 'metrics':
            mode = kwargs.get(key)
            if mode == 'loss':
                label_y = 'Loss'
            if mode == 'IoU' or mode == 'iou':
                label_y == 'IoU'
        if key == 'mode':
            mode = kwargs.get(key)
            if mode == 'train':
                title = 'Training'
            if mode == 'val':
                title = 'Validation'
        if key == 'scale':
            scale = kwargs.get(key)
        if key == 'smoothing':
            smoothing = kwargs.get(key)
    data = [[], []]
    with open(filepath, 'rt') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            if row:
                print(row)
                data[0].append(float(row[0]))
                data[1].append(float(row[1]))
            print(type(row))
        y_data = moving_average(data[1], len(data[1]) * smoothing)
        x_data = data[0]
        plt.plot(x_data, y_data)
        plt.xlabel(label_x)
        plt.ylabel(label_y)
        plt.yscale(scale)
        plt.title(title)
        plt.show()
        print('done')
# TODO : Faire marcher cette ****** de fonction de visualisation de labels...
def lab2color(lab, axis):
    """
    A simple helper function which computes a color, given a label
    :param lab: the input label
    :type lab: a 1D, one-elemenent np array
    :return: color, the RGB color of this label
    :rtype: A 1D, 3-element np array
    """
    if lab < 18:
        return np.asarray((13,115,42), dtype=np.uint8)
    else:
        return np.zeros((3))

def convert_labelled_images(image_list=[]):
    """
    Method to visualize easily labelled images by attributing
    :param image_list: paths to the images to process
    :type image_list: an iterable of strings
    :return: nothing, writes images in the same folder as the input images.
    :rtype: None
    """
    for image in image_list:
        if (isfile(image)):
            Im = Image.open(image)
            im = np.asarray(Im, dtype=int)
            new_im = np.expand_dims(im, axis=2)
            new_im = np.apply_over_axes(lab2color, new_im, axes=(0,1))
            print('test')
#
