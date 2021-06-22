import mdtraj as md
from scattering.utils.io import get_fn
from scattering.utils.run import run_total_vhf

def test_run_van_hove():
    trj = md.load(
        get_fn('spce.xtc'),
        top=get_fn('spce.gro')
    )

    chunk_length = 2
    n_chunks = 2

    run_total_vhf(trj, chunk_length, n_chunks)
