import numpy as np
import matplotlib.pyplot as plt
import mdtraj as md
import pytest

from scattering.van_hove import compute_van_hove
from scattering.utils.io import get_fn


@pytest.mark.parametrize("form", ["atomic", "cromer-mann"])
def test_van_hove(form):
    trj = md.load(get_fn("spce.xtc"), top=get_fn("spce.gro"))

    chunk_length = 2

    r, t, g_r_t = compute_van_hove(trj, chunk_length=chunk_length, form=form)

    assert len(t) == 2
    assert len(r) == 200
    assert np.shape(g_r_t) == (2, 200)

    # Check normalization to ~1
    assert 0.95 < np.mean(g_r_t[:, -10:]) < 1.05

    fig, ax = plt.subplots()
    for i in range(len(t)):
        ax.plot(r, g_r_t[i], ".-", label=t[i])
    ax.set_ylim((0, 3))
    ax.legend()
    fig.savefig("vhf.pdf")


def test_serial_van_hove():
    trj = md.load(get_fn("spce.xtc"), top=get_fn("spce.gro"))

    chunk_length = 2

    r, t, g_r_t = compute_van_hove(trj, chunk_length=chunk_length, parallel=False)


def test_van_hove_equal():
    trj = md.load(get_fn("spce.xtc"), top=get_fn("spce.gro"))

    chunk_length = 2

    r_p, t_p, g_r_t_p = compute_van_hove(trj, chunk_length=chunk_length)
    r_s, t_s, g_r_t_s = compute_van_hove(trj, chunk_length=chunk_length, parallel=False)

    assert np.allclose(r_p, r_s)
    assert np.allclose(t_p, t_s)
    assert np.allclose(g_r_t_p, g_r_t_s)
