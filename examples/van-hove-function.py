import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt

from scattering.scattering import compute_van_hove
from scattering.utils.io import get_fn
from scattering.utils.utils import get_dt
from scattering.utils.features import find_local_maxima

trj = md.load(get_fn('10fs.xtc'),
              top=get_fn('10fs.gro'),
              stride=10
              )[:10000]

chunk_length = 5
selection1 = 'name O'
selection2 = 'name O'

r, g_r_t = compute_van_hove(trj, 5)

dt = get_dt(trj)

fig, ax = plt.subplots()

for j in range(5):
    t = round(j * dt, 3)
    r_max, g_r_max = find_local_maxima(r, np.mean(g_r_t[j::chunk_length], axis=0), r_guess=0.3)
    plt.plot(r_max, g_r_max, 'k.')
    r_max, g_r_max = find_local_maxima(r, np.mean(g_r_t[j::chunk_length], axis=0), r_guess=0.45)
    plt.plot(r_max, g_r_max, 'k.')
    plt.plot(r, np.mean(g_r_t[j::chunk_length], axis=0), label='{} ps'.format(t))

ax.set_xlim((0, 0.8))
ax.set_ylim((0, 3.0))
ax.legend(loc=0)
ax.set_xlabel(r'Radial coordinate, $r$, $nm$')
ax.set_ylabel(r'Van Hove Function, , $g(r, t)$, $unitiless$')
plt.savefig('van-hove-function.pdf')
