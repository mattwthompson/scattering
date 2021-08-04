import numpy as np
import mdtraj as md

from scattering.utils.io import get_fn
from scattering.utils.features import find_local_maxima, find_local_minima


def test_local_maxima():
    """ Find maxs and mins for O-O RDF of SPC/E water"""
    data = np.loadtxt(get_fn('rdf.txt'))
    r = data[:,0]
    g_r = data[:,1]

    r_maxes = list()
    for i, r_guess in enumerate([0.3, 0.45, 0.65]):
        r_max, g_r_max = find_local_maxima(r, g_r, r_guess=r_guess)
        r_maxes.append(r_max)

    r_mins = list()
    for i, r_guess in enumerate([0.3, 0.5]):
        r_min, g_r_min = find_local_minima(r, g_r, r_guess=r_guess)
        r_mins.append(r_min)
    print(r_mins)

    assert np.allclose(r_maxes, [0.2725, 0.4475, 0.6775])
    assert np.allclose(r_mins, [0.3325, 0.5625])



def test_local_minimas():
    """Tests find_local_minima for x^2, 2^x, sqrt(x), logit, sin(x)"""
    
    #x^2
    r = np.linspace(-3,3,10000000)
    g_r = r**2
    r_mins =list()
    for r_guess in [-3,0,3]:
        r_min, g_r_min = find_local_minima(r, g_r, r_guess = r_guess)
        r_mins.append(r_min)
    assert np.allclose(r_mins, [0.0, 0.0, 0.0], atol = 0.000001)


    #2^x
    r = np.linspace(-10,10,10000000)
    g_r = 2**r
    r_mins=list()
    for r_guess in [-10,0,10]:
        r_min, g_r_min = find_local_minima(r, g_r, r_guess = r_guess)
        r_mins.append(r_min)
    assert np.allclose(r_mins, [-10, -10, -10], atol = 0.000001)


    #sqrt(x)
    r = np.linspace(0,10,10000000)
    g_r = r**0.5
    r_mins=list()
    for r_guess in [0,5,10]:
        r_min, g_r_min = find_local_minima(r, g_r, r_guess = r_guess)
        r_mins.append(r_min)
    assert np.allclose(r_mins, [0, 0, 0], atol = 0.000001)


    #logit
    from scipy.special import logit
    r = np.linspace(0,10,10000000)
    g_r = logit(r)
    r_mins=list()
    for r_guess in [0,0,10]:
        r_min, g_r_min = find_local_minima(r, g_r, r_guess = r_guess)
        r_mins.append(r_min)
    assert np.allclose(r_mins, [0, 0, 0], atol = 0.000001)


    #sin(x)
    import math
    pi = math.pi
    r = np.linspace(0, 4*pi,1000000)
    g_r = []
    for i in range(len(r)):
        g_r.append(math.sin(r[i]))
        i += 1
    g_r = np.array(g_r)
    r_mins=list()
    for r_guess in [pi,1.6*pi,3*pi]:
        r_min, g_r_min = find_local_minima(r, g_r, r_guess = r_guess)
        r_mins.append(r_min)
    assert np.allclose(r_mins, [1.5*pi, 1.5*pi, 3.5*pi], atol = 0.000001)
