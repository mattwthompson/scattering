import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt

from scattering.scattering import big_vhf_wrapper
from scattering.utils.io import get_fn
from scattering.utils.utils import get_dt

trj = md.load(get_fn('10fs.xtc'),
              top=get_fn('10fs.gro'),
              stride=10
              )[:10000]

r, g_r_t = big_vhf_wrapper(trj, 5, 'name O', 'name O')

dt = get_dt(trj)

fig, ax = plt.subplots()

for j in range(5):
    t = round(j * dt, 3)
    plt.plot(r, np.mean(g_r_t[j::10], axis=0), label='{} ps'.format(t))

ax.set_xlim((0, 0.8))
ax.set_ylim((0, 3.0))
ax.legend(loc=0)
ax.set_xlabel(r'Radial coordinate, $r$, $nm$')
ax.set_ylabel(r'Van Hove Function, , $g(r, t)$, $unitiless$')
plt.savefig('van-hove-function.pdf')
