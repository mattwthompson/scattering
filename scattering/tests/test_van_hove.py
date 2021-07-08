import numpy as np
import matplotlib.pyplot as plt
import mdtraj as md
import pytest

from scattering.van_hove import compute_van_hove
from scattering.utils.io import get_fn
from scattering.van_hove import compute_partial_van_hove
from scattering.van_hove import vhf_from_pvhf
from itertools import combinations_with_replacement
from scattering.utils.constants import get_form_factor
from scattering.van_hove import get_unique_atoms


def test_van_hove():
    trj = md.load(get_fn("spce.xtc"), top=get_fn("spce.gro"))

    chunk_length = 2

    r, t, g_r_t = compute_van_hove(trj, chunk_length=chunk_length)

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


def test_vhf_from_pvhf():
    trj = md.load(get_fn("spce.xtc"), top=get_fn("spce.gro"))
    unique_atoms = get_unique_atoms(trj)
    tuples_combination = combinations_with_replacement(unique_atoms, 2)

    # obtaining g_r_t from total func
    chunk_length = 20
    r, t, g_r_t = compute_van_hove(trj, chunk_length=chunk_length)

    # obtating dict of np.array of pvhf
    partial_dict = {}

    for pairs in tuples_combination:
        pair1 = pairs[0]
        pair2 = pairs[1]
        if pairs[0].name > pairs[1].name:
            pair2 = pairs[0]
            pair1 = pairs[1]
       
        x = compute_partial_van_hove(
            trj,
            chunk_length=chunk_length,
            selection1=f"name {pair1.name}",
            selection2=f"name {pair2.name}",
        )
        partial_dict[pairs] = x[1]

    # obtaining total_grt from partial
    total_g_r_t = vhf_from_pvhf(trj, partial_dict)

    assert np.allclose(g_r_t, total_g_r_t)

trj = md.load(get_fn("spce.xtc"), top=get_fn("spce.gro"))
unique_atoms = get_unique_atoms(trj)
atom = md.core.topology.Atom(name="Na", element=md.core.element.sodium, index=0, residue=1)

@pytest.mark.parametrize(
    "tuple_keys", [(unique_atoms[0], unique_atoms[1], unique_atoms[1]), ("H","O"),("H-O"),"HO", (atom, unique_atoms[0])]
)
def test_pvhf_error(tuple_keys):
    trj = md.load(get_fn("spce.xtc"), top=get_fn("spce.gro"))

    topology = trj.topology
    atoms = [i.element.symbol for i in topology.atoms]
    atom_list = sorted(set(atoms))
    chunk_length = 20

    # obtating dict of np.array of pvhf
    partial_dict = {}
    x = compute_partial_van_hove(
        trj,
        chunk_length=chunk_length,
        selection1=f"name O",
        selection2=f"name O",
    )
    partial_dict[tuple_keys] = x[1]

    with pytest.raises(ValueError):
        vhf_from_pvhf(trj, partial_dict)

