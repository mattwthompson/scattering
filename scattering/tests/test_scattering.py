import numpy as np
import matplotlib.pyplot as plt
import mdtraj as md
import pytest

from scattering.scattering import structure_factor
from scattering.utils.io import get_fn


@pytest.mark.parametrize("weighting_factor", ["fz", "al"])
def test_structure_factor(weighting_factor):
    trj = md.load(get_fn("spce.xtc"), top=get_fn("spce.gro"))[:100]

    Q, S = structure_factor(
        trj, Q_range=(0.5, 200), framewise_rdf=False, weighting_factor=weighting_factor
    )


@pytest.mark.parametrize("form", ["atomic", "cromer-mann"])
def test_sq_form(form):
    trj = md.load(get_fn("spce.xtc"), top=get_fn("spce.gro"))[:100]

    Q, S = structure_factor(trj, Q_range=(0.5, 200), framewise_rdf=False, form=form)


def test_invalid_form_style():
    trj = md.load(get_fn("spce.xtc"), top=get_fn("spce.gro"))[:100]

    with pytest.raises(ValueError):
        structure_factor(trj, Q_range=(0.5, 200), framewise_rdf=False, form="random")


def test_invalid_sq_weigting_factor():
    trj = md.load(get_fn("spce.xtc"), top=get_fn("spce.gro"))[:100]

    with pytest.raises(ValueError):
        Q, S = structure_factor(
            trj, Q_range=(0.5, 200), framewise_rdf=False, weighting_factor="invalid"
        )

def test_with_master():
    trj = md.load(get_fn("spce.xtc"), top=get_fn("spce.gro"))[:100]

    Q, S = structure_factor(
        trj, Q_range=(0.5, 200), framewise_rdf=False, weighting_factor="fz"
    )

    master_Q = np.loadtxt(get_fn("Q.txt"))
    master_S = np.loadtxt(get_fn("S.txt"))

    assert np.allclose(Q, master_Q)
    assert np.allclose(S, master_S, atol=1e-2)
