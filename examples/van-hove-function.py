import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt

from scattering.scattering import compute_van_hove
from scattering.utils.io import get_fn
from scattering.utils.utils import get_dt
from scattering.utils.features import find_local_maxima

trj = md.load(get_fn('10fs.xtc'),
              top=get_fn('10fs.gro'),
              stride=1,
              )[:1000]

chunk_length = 200
selection1 = 'name O'
selection2 = 'name O'

r, t, g_r_t = compute_van_hove(trj, chunk_length)

dt = get_dt(trj)

# Save output to text files
np.savetxt('vhf.txt', g_r_t, header='# Van Hove Function, dt: {} fs, dr: {}'.format(
    dt,
    np.unique(np.round(np.diff(trj.time), 6))[0],
))
np.savetxt('r.txt', r)
np.savetxt('t.txt', t)


fig, ax = plt.subplots()

for j in range(chunk_length):
    r_max, g_r_max = find_local_maxima(r, g_r_t[j], r_guess=0.3)
    plt.plot(r_max, g_r_max, 'k.')
    r_max, g_r_max = find_local_maxima(r, g_r_t[j], r_guess=0.45)
    plt.plot(r_max, g_r_max, 'k.')
    plt.plot(r, g_r_t[j], label='{} ps'.format(t[j]))


# Save output to text files
np.savetxt('vhf.txt', g_r_t)
np.savetxt('r.txt', r)
np.savetxt('t.txt', t)


ax.set_xlim((0, 0.8))
ax.set_ylim((0, 3.0))
ax.legend(loc=0)
ax.set_xlabel(r'Radial coordinate, $r$, $nm$')
ax.set_ylabel(r'Van Hove Function, , $g(r, t)$, $unitiless$')
plt.savefig('van-hove-function.pdf')


fig, ax = plt.subplots()

heatmap = ax.imshow(g_r_t[::-1]-1, vmin=-0.04, vmax=0.04, cmap='RdYlGn_r')
plt.colorbar(heatmap)

fig.savefig('heatmap.pdf')

# Save output to text files
np.savetxt('vhf.txt', g_r_t)
np.savetxt('r.txt', r)
np.savetxt('t.txt', t)
