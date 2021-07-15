import numpy as np
import mdtraj as md
import pytest

from scattering.utils.io import get_fn
from scattering.utils.run import run_total_vhf, run_partial_vhf


@pytest.mark.parametrize("step", [1, 2])
def test_run_total_vhf(step):
    trj = md.load(get_fn("spce.xtc"), top=get_fn("spce.gro"))

    chunk_length = 4
    n_chunks = 5

    r, t, g_r_t = run_total_vhf(
        trj, step=step, chunk_length=chunk_length, n_chunks=n_chunks
    )

    assert len(t) == chunk_length / step
    assert len(r) == 200
    assert np.shape(g_r_t) == (chunk_length / step, 200)

    # Check normalization to ~1
    assert 0.95 < np.mean(g_r_t[:, -10:]) < 1.05


@pytest.mark.parametrize("step", [1, 2])
def test_run_partial_vhf(step):
    trj = md.load(get_fn("spce.xtc"), top=get_fn("spce.gro"))

    chunk_length = 4
    n_chunks = 5

    combo = ["O", "O"]
    r, t, g_r_t = run_partial_vhf(
        trj,
        step=step,
        selection1=f"element {combo[0]}",
        selection2=f"element {combo[1]}",
        chunk_length=chunk_length,
        n_chunks=n_chunks,
    )

    assert len(t) == chunk_length / step
    assert len(r) == 200
    assert np.shape(g_r_t) == (chunk_length / step, 200)

    # Check normalization to ~1
    assert 0.95 < np.mean(g_r_t[:, -10:]) < 1.05
