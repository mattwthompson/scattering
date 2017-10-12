import itertools as it

import mdtraj as md
import numpy as np
from scipy.integrate import simps


__all__ = ['structure_factor']

def structure_factor(trj, Q_range=(1, 100)):
    """Compute the structure factor.

    The consdered trajectory must include valid elements.

    The computed structure factor is only valid for certain values of Q. The
    lowest value of Q that can sufficiently be described by a box of
    characteristic length `L` is `2 * pi / (L / 2)`.

    Parameters
    ----------
    trj : mdtraj.Trajectory
        A trajectory for which the structure factor is to be computed.
    Q_range : tuple
        The minimum and maximum scattering vectors, in `1/nm`, to be consdered.

    Returns
    -------
    Q : np.ndarray
        The values of the scattering vector, in `1/nm`, that was considered.
    S : np.ndarray
        The structure factor of the trajectory

    """
    rho = np.mean(trj.n_atoms / trj.unitcell_volumes)
    L = np.min(trj.unitcell_lengths)
    Q = np.logspace(np.log10(Q_range[0]), np.log10(Q_range[1]))
    elements = set([a.element for a in trj.topology.atoms])

    compositions = dict()
    form_factors = dict()
    rdfs = dict()

    S = np.zeros(shape=(len(Q)))

    for element in elements:
        compositions[element.symbol] = len(trj.topology.select('element {}'.format(element.symbol)))/trj.n_atoms
        form_factors[element.symbol] = element.atomic_number

    for i,q in enumerate(Q):
        num = 0
        denom = 0
        for element in elements:
            denom += (compositions[element.symbol] * form_factors[element.symbol]) **2

        for (element1, element2) in it.combinations_with_replacement(elements, 2):
            e1 = element1.symbol
            e2 = element2.symbol

            f_a = form_factors[e1]
            f_b = form_factors[e2]

            x_a = compositions[e1]
            x_b = compositions[e2]

            pre_factor = x_a * x_b * f_a * f_b * 4 * np.pi * rho
            try:
                g_r = rdfs['{0}{1}'.format(e1, e2)]
            except KeyError:
                r, g_r = md.compute_rdf(trj,
                                       pairs=trj[0].topology.select_pairs(selection1='element {}'.format(e1),
                                                                          selection2='element {}'.format(e2)),
                                       r_range=(0, L / 2),
                                       bin_width=0.001)
                rdfs['{0}{1}'.format(e1, e2)] = g_r
            integral = simps(r ** 2 * (g_r - 1) * np.sin(q * r) / (q * r), r)
            num += pre_factor * integral + int(e1 == e2)
        S[i] = num/denom
    return Q, S
