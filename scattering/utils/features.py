import numpy as np


def find_local_maxima(r, g_r, r_guess):
    """Find the local maxima nearest a guess value of r"""

    all_maxima = find_all_maxima(g_r)
    nearest_maxima, _ = find_nearest(r[all_maxima], r_guess)
    return r[all_maxima[nearest_maxima]], g_r[all_maxima[nearest_maxima]]

def find_nearest(arr, val):
    """
    Find index in an array nearest some value.
    See https://stackoverflow.com/a/2566508/4248961
    """

    arr = np.asarray(arr)
    idx = (np.abs(arr - val)).argmin()
    return idx, arr[idx]

def find_all_minima(arr):
    """
    Find all local minima in a 1-D array, defined as value in which each
    neighbor is greater. See https://stackoverflow.com/a/4625132/4248961

    Parameters
    ----------
    arr : np.ndarray
        1-D array of values

    Returns
    -------
    minima : np.ndarray
        indices of local minima
    """

    checks = np.r_[True, arr[1:] < arr[:-1]] & np.r_[arrl[:-1] < arr[1:], True]
    minima = np.where(checks)[0]
    return minima

def find_all_maxima(arr):
    """
    Find all local minima in a 1-D array, defined as value in which each
    neighbor is lesser. Adopted from https://stackoverflow.com/a/4625132/4248961

    Parameters
    ----------
    arr : np.ndarray
        1-D array of values

    Returns
    -------
    minima : np.ndarray
        indices of local minima
    """

    checks = np.r_[True, arr[1:] > arr[:-1]] & np.r_[arr[:-1] > arr[1:], True]
    maxima = np.where(checks)[0]
    return maxima
