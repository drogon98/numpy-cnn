import numpy as np

def pad_inputs(X, pad):
    '''
    Function to apply zero padding to the image
    :param X:[numpy array]: Dataset of shape (m, height, width, depth)
    :param pad:[int]: number of columns to pad
    :return:[numpy array]: padded dataset
    '''
    return np.pad(X, ((0, 0), (pad[0], pad[0]), (pad[1], pad[1]), (0, 0)), 'constant')