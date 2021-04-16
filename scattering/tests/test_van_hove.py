import numpy as np
import matplotlib.pyplot as plt
import mdtraj as md
import pytest

from scattering.van_hove import compute_van_hove
from scattering.utils.io import get_fn

@pytest.mark.parametrize("self_correlation", [True, False, "self"])
def test_van_hove(self_correlation):
    trj = md.load(
        get_fn('spce.xtc'),
        top=get_fn('spce.gro')
    )[:100]

    chunk_length = 2

    r, t, g_r_t = compute_van_hove(trj, 
                                   self_correlation=self_correlation,
                                   chunk_length=chunk_length)

    assert len(t) == 2
    assert len(r) == 200
    assert np.shape(g_r_t) == (2, 200)

    # Check normalization to ~1
    assert 0.95 < np.mean(g_r_t[:, 100:]) < 1.05

    fig, ax = plt.subplots()
    for i in range(len(t)):
        ax.plot(r, g_r_t[i], '.-', label=t[i])
    ax.set_ylim((0, 3))
    ax.legend()
    fig.savefig('vhf.pdf')
