import numpy as np
import pytest
import matplotlib.pyplot as plt
import mdtraj as md
import pytest

from scattering.van_hove import (
    compute_van_hove,
    compute_partial_van_hove,
    compute_partial_van_hove,
    vhf_from_pvhf,
)
from scattering.utils.io import get_fn
from itertools import combinations_with_replacement
from scattering.utils.constants import get_form_factor
from scattering.utils.utils import get_unique_atoms


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


# @pytest.mark.parametrize("self_correlation", [True, False, "self"])
# def test_van_hove_self(self_correlation):
#    trj = md.load(get_fn("spce.xtc"), top=get_fn("spce.gro"))
#    chunk_length = 2
#
#    r, t, g_r_t = compute_van_hove(trj,
#                                   self_correlation=self_correlation,
#                                   chunk_length=chunk_length)
#
#    assert len(t) == 2
#    assert len(r) == 200
#    assert np.shape(g_r_t) == (2, 200)


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


def test_self_partial_warning():
    trj = md.load(get_fn("spce.xtc"), top=get_fn("spce.gro"))

    chunk_length = 2

    with pytest.warns(UserWarning, match=r"Partial VHF"):
        compute_partial_van_hove(
            trj,
            chunk_length=chunk_length,
            selection1="name O",
            selection2="name H",
            self_correlation=True,
        )


@pytest.mark.parametrize("parallel", [True, False])
def test_self_warning(parallel):
    trj = md.load(get_fn("spce.xtc"), top=get_fn("spce.gro"))

    chunk_length = 2

    with pytest.warns(UserWarning, match=r"Total VHF"):
        compute_van_hove(
            trj,
            chunk_length=chunk_length,
            self_correlation=True,
            parallel=parallel,
        )


def test_vhf_from_pvhf():
    trj = md.load(get_fn("spce.xtc"), top=get_fn("spce.gro"))
    unique_atoms = get_unique_atoms(trj)
    tuples_combination = combinations_with_replacement(unique_atoms, 2)

    # obtaining g_r_t from total func
    chunk_length = 20
    r, t, g_r_t = compute_van_hove(trj, chunk_length=chunk_length)
    partial_dict = compute_van_hove(trj, chunk_length=chunk_length, partial=True)
    # obtating dict of np.array of pvhf
    partial_dict = {}

    for pairs in tuples_combination:
        pair1 = pairs[0]
        pair2 = pairs[1]
        # Set in alphabetical order
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


def test_pvhf_error_2_atoms_per_pair():
    trj = md.load(get_fn("spce.xtc"), top=get_fn("spce.gro"))
    unique_atoms = get_unique_atoms(trj)

    partial_dict = {}
    x = compute_partial_van_hove(
        trj,
        chunk_length=20,
        selection1="name O",
        selection2="name O",
    )
    partial_dict[(unique_atoms[0], unique_atoms[1], unique_atoms[1])] = x[1]

    with pytest.raises(
        ValueError, match="Dictionary key not valid. Must only have 2 atoms per pair"
    ):
        vhf_from_pvhf(trj, partial_dict)


def test_pvhf_error_atoms_in_trj():
    trj = md.load(get_fn("spce.xtc"), top=get_fn("spce.gro"))
    unique_atoms = get_unique_atoms(trj)
    atom = md.core.topology.Atom(
        name="Na", element=md.core.element.sodium, index=0, residue=1
    )

    partial_dict = {}
    x = compute_partial_van_hove(
        trj,
        chunk_length=20,
        selection1="name O",
        selection2="name O",
    )
    partial_dict[(atom, unique_atoms[0])] = x[1]

    with pytest.raises(ValueError, match="Dictionary key not valid, `Atom`"):
        vhf_from_pvhf(trj, partial_dict)


def test_pvhf_error_is_atom_type():
    trj = md.load(get_fn("spce.xtc"), top=get_fn("spce.gro"))
    unique_atoms = get_unique_atoms(trj)

    partial_dict = {}
    x = compute_partial_van_hove(
        trj,
        chunk_length=20,
        selection1="name O",
        selection2="name O",
    )
    partial_dict[("H", "O")] = x[1]

    with pytest.raises(ValueError, match="Dictionary key not valid. Must be type"):
        vhf_from_pvhf(trj, partial_dict)


def test_pvhf_error_is_tuple():
    trj = md.load(get_fn("spce.xtc"), top=get_fn("spce.gro"))
    unique_atoms = get_unique_atoms(trj)
    key = frozenset({unique_atoms[0], unique_atoms[1]})
    partial_dict = {}
    x = compute_partial_van_hove(
        trj,
        chunk_length=20,
        selection1="name O",
        selection2="name O",
    )
    partial_dict[key] = x[1]

    with pytest.raises(ValueError, match="Dictionary key not valid. Must be a tuple"):
        vhf_from_pvhf(trj, partial_dict)


def test_self_partial_error():
    trj = md.load(get_fn("spce.xtc"), top=get_fn("spce.gro"))

    chunk_length = 2

    with pytest.raises(ValueError):
        compute_partial_van_hove(
            trj,
            chunk_length=chunk_length,
            selection1="name O",
            selection2="name H",
            self_correlation="self",
            parallel=True,
        )


@pytest.mark.parametrize(
    ("self_correlation", "normalization"),
    [(True, (0.95, 1.05)), (False, (0.95, 1.05)), ("self", (-0.05, 0.05))],
)
def test_partial_van_hove(self_correlation, normalization):
    trj = md.load(get_fn("spce.xtc"), top=get_fn("spce.gro"))[:100]
    chunk_length = 2

    r, g_r_t = compute_partial_van_hove(
        trj,
        selection1="name O",
        selection2="name O",
        self_correlation=self_correlation,
        parallel=False,
        chunk_length=chunk_length,
    )

    assert len(r) == 200
    assert np.shape(g_r_t) == (2, 200)

    # Check normalization
    assert normalization[0] < np.mean(g_r_t[:, -10:]) < normalization[1]
