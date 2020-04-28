import numpy as np
import matplotlib.pyplot as plt
import mdtraj as md

from scattering.scattering import structure_factor
from scattering.utils.io import get_fn

def test_structure_factor():
    trj = md.load(
        get_fn('spce.xtc'),
        top=get_fn('spce.gro')
    )[:100]

    Q, S = structure_factor(temp_trj,
                            Q_range=(0.5, 200),
                            framewise_rdf=False,
                            method='fz')
