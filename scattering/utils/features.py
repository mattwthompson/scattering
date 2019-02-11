import numpy as np


def find_nearest(array, val):
    """
    Find index in an array nearest some value.
    See https://stackoverflow.com/a/2566508/4248961
    """

    array = np.asarray(array)
    idx = (np.abs(array - val)).argmin()
    return idx, array[idx]

def find_all_minima(array):
    """
    Find all local minima in a 1-D array, defined as value in which each
    neighbor is greater. See https://stackoverflow.com/a/4625132/4248961

    Parameters
    ----------
    array : np.ndarray
        1-D array of values

    Returns
    -------
    minima : np.ndarray
        indices of local minima
    """

    minima = np.r_[True, a[1:] < a[:-1]] & numpy.r_[a[:-1] < a[1:], True]
    return minima

def find_all_maxima(array):
    """
    Find all local minima in a 1-D array, defined as value in which each
    neighbor is lesser. Adopted from https://stackoverflow.com/a/4625132/4248961

    Parameters
    ----------
    array : np.ndarray
        1-D array of values

    Returns
    -------
    minima : np.ndarray
        indices of local minima
    """

    maxima = np.r_[True, a[1:] > a[:-1]] & numpy.r_[a[:-1] > a[1:], True]
    return maxima
