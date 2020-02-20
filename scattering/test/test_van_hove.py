import numpy as np
import matplotlib.pyplot as plt
import mdtraj as md

from scattering.van_hove import compute_van_hove
from scattering.utils.io import get_fn


trj = md.load(
    get_fn('spce.xtc'),
    top=get_fn('spce.gro')
)

fig, ax = plt.subplots()
r, t, g_r_t = compute_van_hove(trj, chunk_length=2)
for i in range(len(t)):
    ax.plot(r, g_r_t[i], '.-', label=t[i])
ax.set_ylim((0, 3))
ax.legend()
fig.savefig('vhf.pdf')
