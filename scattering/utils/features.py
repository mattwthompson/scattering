import numpy as np


def find_local_maxima(r, g_r, r_guess):
    """Find the local maxima nearest a guess value of r"""

    all_maxima = find_all_maxima(g_r)
    nearest_maxima, _ = find_nearest(r[all_maxima], r_guess)
    return r[all_maxima[nearest_maxima]], g_r[all_maxima[nearest_maxima]]

def find_local_minima(r, g_r, r_guess):
    """Find the local minima nearest a guess value of r"""

    all_minima = find_all_minima(g_r)
    nearest_minima, _ = find_nearest(r[all_minima], r_guess)
    return r[all_minima[nearest_minima]], g_r[all_minima[nearest_minima]]

def maxima_in_range(r, g_r, r_min, r_max):
    """Find the maxima in a range of r, g_r values"""
    idx = np.where(np.logical_and(np.greater_equal(r, r_min), np.greater_equal(r_max, r)))
    g_r_slice = g_r[idx]
    g_r_max = g_r_slice[g_r_slice.argmax()]
    idx_max, _ = find_nearest(g_r, g_r_max)
    return r[idx_max], g_r[idx_max]

def minima_in_range(r, g_r, r_min, r_max):
    """Find the minima in a range of r, g_r values"""
    idx = np.where(np.logical_and(np.greater_equal(r, r_min), np.greater_equal(r_max, r)))
    g_r_slice = g_r[idx]
    g_r_min = g_r_slice[g_r_slice.argmin()]
    idx_min, _ = find_nearest(g_r, g_r_min)
    return r[idx_min], g_r[idx_min]

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

    checks = np.r_[True, arr[1:] < arr[:-1]] & np.r_[arr[:-1] < arr[1:], True]
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
