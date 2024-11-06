import numpy as np


def central_difference_x(field, element_length):
    diff = np.zeros_like(field)
    diff[1:-1, 1:-1] = (field[1:-1, 2:] - field[1:-1, :-2]) / (2. * element_length)
    return diff


def central_difference_y(field, element_length):
    diff = np.zeros_like(field)
    diff[1:-1, 1:-1] = (field[2:, 1:-1] - field[:-2, 1:-1]) / (2. * element_length)
    return diff


def laplace(field, element_length):
    diff = np.zeros_like(field)
    diff[1:-1, 1:-1] = field[1:-1, :-2] + field[:-2, 1:-1] - 4 * field[1:-1, 1:-1] + field[1:-1, 2:] + field[2:, 1:-1]
    diff = diff / (element_length ** 2)
    return diff
