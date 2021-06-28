import numpy as np
import matplotlib.pyplot as plt
import mdtraj as md

from scattering.van_hove import compute_van_hove
from scattering.utils.io import get_fn
from scattering.van_hove import compute_partial_van_hove
from scattering.van_hove import vhf_from_pvhf
from scattering.van_hove import atom_conc_from_list
from scattering.van_hove import form_factor_from_list
from itertools import combinations_with_replacement
from scattering.utils.constants import get_form_factor



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

    topology = trj.topology
    atoms = [i.element.symbol for i in topology.atoms]
    atom_list = sorted(set(atoms))

    # obtaining g_r_t from total func
    chunk_length = 20
    r, t, g_r_t = compute_van_hove(trj, chunk_length=chunk_length)

    # obtating dict of np.array of pvhf
    partial_dict = {}
    combination = list(combinations_with_replacement(atom_list, 2))
    for pairs in combination:
        x = compute_partial_van_hove(
            trj,
            chunk_length=chunk_length,
            selection1=f"name {pairs[0]}",
            selection2=f"name {pairs[1]}",
        )
        partial_dict[f"{pairs[0]}{pairs[1]}"] = x[1]

    # obtaining total_grt from partial
    total_g_r_t = vhf_from_pvhf(trj, partial_dict, atom_list)

    assert np.allclose(g_r_t, total_g_r_t)
