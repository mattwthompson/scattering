import numpy as np
import matplotlib.pyplot as plt

from scattering.utils.features import *
from scattering.utils.io import get_fn

#  Load data
data = np.loadtxt(get_fn('rdf.txt'))
r = data[:, 0]
g_r = data[:, 1]

fig, ax = plt.subplots()
ax.plot(r, g_r, '-')

for i, r_guess in enumerate([0.3, 0.45, 0.65]):
    r_max, g_r_max = find_local_maxima(r, g_r, r_guess=r_guess)
    plt.plot(r_max, g_r_max, 'k.')

for i, r_guess in enumerate([0.3, 0.5]):
    r_min, g_r_min = find_local_minima(r, g_r, r_guess=r_guess)
    plt.plot(r_min, g_r_min, 'r.')

ax.set_xlabel(r'Radial coordinate, $r$, $nm$')
ax.set_ylabel(r'Radial distribution function, $g(r)$, $unitiless$')
plt.savefig('maxima-minima.pdf')
