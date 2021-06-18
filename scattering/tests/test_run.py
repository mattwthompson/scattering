import numpy as np
import mdtraj as md

from scattering.utils.io import get_fn
from scattering.utils.run import run_total_vhf, run_partial_vhf

def test_run_total_vhf():
    trj = md.load(
        get_fn('spce.xtc'),
        top=get_fn('spce.gro')
    )

    chunk_length = 2
    n_chunks=5

    r, t, g_r_t = run_total_vhf(trj, chunk_length=chunk_length, n_chunks=n_chunks)

    assert len(t) == 2
    assert len(r) == 200
    assert np.shape(g_r_t) == (2, 200)

    # Check normalization to ~1
    assert 0.95 < np.mean(g_r_t[:, -10:]) < 1.05

def test_run_partial_vhf():
    trj = md.load(
        get_fn('spce.xtc'),
        top=get_fn('spce.gro')
    )

    chunk_length = 2
    n_chunks=5

    combo = ["O", "O"]
    r, t, g_r_t = run_partial_vhf(trj, selection1=combo[0], selection2=combo[1], chunk_length=chunk_length, n_chunks=n_chunks)

    assert len(t) == 2
    assert len(r) == 200
    assert np.shape(g_r_t) == (2, 200)

    # Check normalization to ~1
    assert 0.95 < np.mean(g_r_t[:, -10:]) < 1.05
