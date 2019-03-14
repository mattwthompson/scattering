import itertools as it

import mdtraj as md
import numpy as np
from scipy.integrate import simps

from scattering.utils.utils import rdf_by_frame
from scattering.utils.utils import get_dt
from scattering.utils.constants import get_form_factor


__all__ = ['structure_factor', 'compute_partial_van_hove', 'compute_van_hove']



def structure_factor(trj, Q_range=(0.5, 50), n_points=1000, framewise_rdf=False):
    """Compute the structure factor.

    The consdered trajectory must include valid elements.

    The computed structure factor is only valid for certain values of Q. The
    lowest value of Q that can sufficiently be described by a box of
    characteristic length `L` is `2 * pi / (L / 2)`.

    Parameters
    ----------
    trj : mdtraj.Trajectory
        A trajectory for which the structure factor is to be computed.
    Q_range : list or np.ndarray, default=(0.5, 50)
        Minimum and maximum Values of the scattering vector, in `1/nm`, to be
        consdered.
    n_points : int, default=1000
    framewise_rdf : boolean, default=False
        If True, computes the rdf frame-by-frame. This can be useful for
        managing memory in large systems.

    Returns
    -------
    Q : np.ndarray
        The values of the scattering vector, in `1/nm`, that was considered.
    S : np.ndarray
        The structure factor of the trajectory

    """
    rho = np.mean(trj.n_atoms / trj.unitcell_volumes)
    L = np.min(trj.unitcell_lengths)

    top = trj.topology
    elements = set([a.element for a in top.atoms])

    compositions = dict()
    form_factors = dict()
    rdfs = dict()

    Q = np.logspace(np.log10(Q_range[0]),
                    np.log10(Q_range[1]),
                    num=n_points)
    S = np.zeros(shape=(len(Q)))

    for elem in elements:
        compositions[elem.symbol] = len(top.select('element {}'.format(elem.symbol)))/trj.n_atoms
        form_factors[elem.symbol] = elem.atomic_number

    for i, q in enumerate(Q):
        num = 0
        denom = 0
        for elem in elements:
            denom += (compositions[elem.symbol] * form_factors[elem.symbol]) ** 2

        for (elem1, elem2) in it.combinations_with_replacement(elements, 2):
            e1 = elem1.symbol
            e2 = elem2.symbol

            f_a = form_factors[e1]
            f_b = form_factors[e2]

            x_a = compositions[e1]
            x_b = compositions[e2]

            pre_factor = x_a * x_b * f_a * f_b * 4 * np.pi * rho
            try:
                g_r = rdfs['{0}{1}'.format(e1, e2)]
            except KeyError:
                pairs = top.select_pairs(selection1='element {}'.format(e1),
                                         selection2='element {}'.format(e2))
                if framewise_rdf:
                    r, g_r = rdf_by_frame(trj,
                                         pairs=pairs,
                                         r_range=(0, L / 2),
                                         bin_width=0.001)
                else:
                    r, g_r = md.compute_rdf(trj,
                                            pairs=pairs,
                                            r_range=(0, L / 2),
                                            bin_width=0.001)
                rdfs['{0}{1}'.format(e1, e2)] = g_r
            integral = simps(r ** 2 * (g_r - 1) * np.sin(q * r) / (q * r), r)
            num += pre_factor * integral + int(e1 == e2)
        S[i] = num/denom
    return Q, S

def compute_dynamic_rdf(trj):
    """Compute r_ij(t), the distance between atom j at time t and atom i and
    time 0. Note that this alone is likely useless, but is an intermediate
    variable in the construction of a dynamic structure factor. 
    See 10.1103/PhysRevE.59.623.

    Parameters
    ----------
    trj : mdtraj.Trajectory
        A trajectory for which the structure factor is to be computed

    Returns
    -------
    r_ij : np.ndarray, shape=(trj.n_atoms, trj.n_atoms, trj.n_frames)
        A three-dimensional array of interatomic distances
    """

    n_atoms = trj.n_atoms
    n_frames = trj.n_frames

    r_ij = np.ndarray(shape=(trj.n_atoms, trj.n_atoms, trj.n_frames))

    for n_frame, frame in enumerate(trj):
        for atom_i in range(trj.n_atoms):
            for atom_j in range(trj.n_atoms):
                r_ij[atom_i, atom_j, n_frame] = compute_distance(trj.xyz[n_frame, atom_j], trj.xyz[0, atom_i])

    return r_ij

def compute_distance(point1, point2):
    return np.sqrt(np.sum((point1 -point2) ** 2))


def compute_van_hove(trj, chunk_length, water=False,
                     r_range=(0, 1.0), bin_width=0.005, n_bins=None,
                     periodic=True, opt=True):
    """Compute the partial van Hove function of a trajectory

    Parameters
    ----------
    trj : mdtraj.Trajectory
        trajectory on which to compute the Van Hove function
    chunk_length : int
        length of time between restarting averaging
    water : bool
        use X-ray form factors for water that account for polarization
    r_range : array-like, shape=(2,), optional, default=(0.0, 1.0)
        Minimum and maximum radii.
    bin_width : float, optional, default=0.005
        Width of the bins in nanometers.
    n_bins : int, optional, default=None
        The number of bins. If specified, this will override the `bin_width`
         parameter.

    Returns
    -------
    r : numpy.ndarray
        r positions generated by histogram binning
    g_r_t : numpy.ndarray
        Van Hove function at each time and position
    """

    unique_elements = list(set([a.element for a in trj.top.atoms]))

    norm = 0
    g_r_t = None

    for elem1, elem2 in it.combinations_with_replacement(unique_elements[::-1], 2):
        r, g_r_t_partial = compute_partial_van_hove(trj=trj,
                                                    chunk_length=chunk_length,
                                                    selection1='element {}'.format(elem1.symbol),
                                                    selection2='element {}'.format(elem2.symbol),
                                                    r_range=r_range,
                                                    bin_width=bin_width,
                                                    n_bins=n_bins,
                                                    periodic=periodic,
                                                    opt=opt)


        concentration1 = trj.atom_slice(trj.top.select('element {}'.format(elem1.symbol))).n_atoms / trj.n_atoms
        concentration2 = trj.atom_slice(trj.top.select('element {}'.format(elem2.symbol))).n_atoms / trj.n_atoms
        form_factor1 = get_form_factor(element_name=elem1.symbol, water=water)
        form_factor2 = get_form_factor(element_name=elem2.symbol, water=water)

        coeff = form_factor1 * concentration1 * form_factor2 * concentration2

        if g_r_t is None:
            g_r_t = np.zeros_like(g_r_t_partial)
        g_r_t += g_r_t_partial * coeff

        norm += coeff

    # Reshape g_r_t to better represent the discretization in both r and t
    g_r_t_final = np.empty(shape=(chunk_length, len(r)))
    for i in range(chunk_length):
        g_r_t_final[i, :] = np.mean(g_r_t[i::chunk_length], axis=0)

    g_r_t_final /= norm

    t = trj.time[:chunk_length]

    return r, t, g_r_t_final


def compute_partial_van_hove(trj, chunk_length=10, selection1=None, selection2=None,
                             r_range=(0, 0.1), bin_width=0.005, n_bins=200,
                             periodic=True, opt=True):
    """Compute the partial van Hove function of a trajectory

    Parameters
    ----------
    trj : mdtraj.Trajectory
        trajectory on which to compute the Van Hove function
    chunk_length : int
        length of time between restarting averaging
    selection1 : str
        selection to be considered, in the style of MDTraj atom selection
    selection2 : str
        selection to be considered, in the style of MDTraj atom selection
    r_range : array-like, shape=(2,), optional, default=(0.0, 1.0)
        Minimum and maximum radii.
    bin_width : float, optional, default=0.005
        Width of the bins in nanometers.
    n_bins : int, optional, default=None
        The number of bins. If specified, this will override the `bin_width`
         parameter.

    Returns
    -------
    r : numpy.ndarray
        r positions generated by histogram binning
    g_r_t : numpy.ndarray
        Van Hove function at each time and position
    """

    unique_elements = (
        set([a.element for a in trj.atom_slice(trj.top.select(selection1)).top.atoms]),
        set([a.element for a in trj.atom_slice(trj.top.select(selection2)).top.atoms]),
    )

    if any([len(val) > 1 for val in unique_elements]):
        raise UserWarning(
            'Multiple elements found in a selection(s). Results may not be '
            'direcitly comprable to scattering experiments.'
        )

    # Don't need to store it, but this serves to check that dt is constant
    dt = get_dt(trj)

    pairs = trj.top.select_pairs(selection1=selection1, selection2=selection2)

    n_chunks = int(trj.n_frames / chunk_length)

    g_r_t = None
    for i in range(n_chunks):
        times = list()
        for j in range(chunk_length):
            times.append([chunk_length*i, chunk_length*i+j])
        r, g_r_t_frame = md.compute_rdf_t(trj, pairs=pairs, times=times,
                                          r_range=r_range, bin_width=bin_width, n_bins=n_bins,
                                          periodic=periodic, opt=opt)
        if g_r_t is None:
            g_r_t = np.zeros_like(g_r_t_frame)
        g_r_t += g_r_t_frame
    g_r_t /= n_chunks

    return r, g_r_t
