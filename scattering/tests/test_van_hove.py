import numpy as np
import matplotlib.pyplot as plt
import mdtraj as md
import pytest

from scattering.van_hove import compute_van_hove, compute_2d_van_hove
from scattering.utils.io import get_fn

def test_van_hove():
    trj = md.load(
        get_fn('spce.xtc'),
        top=get_fn('spce.gro')
    )[:100]

    chunk_length = 2

    r, t, g_r_t = compute_van_hove(trj, chunk_length=chunk_length)

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

def test_2d_van_hove():
    trj = md.load(
        get_fn('spce.xtc'),
        top=get_fn('spce.gro')
    )[:100]

    chunk_length = 2

    r, t, g_r_t = compute_2d_van_hove(trj, chunk_length=chunk_length)

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
    fig.savefig('2d_vhf.pdf')

@pytest.mark.parametrize('cutoff', [2, [0, 2, 4]])
def test_2d_cutoff(cutoff):
    trj = md.load(
        get_fn('spce.xtc'),
        top=get_fn('spce.gro')
    )[:100]

    chunk_length = 2
    with pytest.raises(ValueError):
        r, t, g_r_t = compute_2d_van_hove(trj, chunk_length=chunk_length, cutoff=cutoff)

@pytest.mark.parametrize('coords', [[True, True, True], [False, False, False]])
def test_2d_coords(coords):
    trj = md.load(
        get_fn('spce.xtc'),
        top=get_fn('spce.gro')
    )[:100]

    chunk_length = 2
    with pytest.raises(ValueError):
        r, t, g_r_t = compute_2d_van_hove(trj, chunk_length=chunk_length, coords=coords)
