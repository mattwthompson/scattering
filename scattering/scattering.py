import itertools as it
from progressbar import ProgressBar

import mdtraj as md
import numpy as np
from scipy.integrate import simps

from scattering.utils.utils import rdf_by_frame
from scattering.utils.utils import get_dt
from scattering.utils.constants import get_form_factor


#__all__ = ['structure_factor', 'compute_partial_van_hove', 'compute_van_hove']



def structure_factor(trj, pair=None, Q_range=(0.5, 50), n_points=1000, framewise_rdf=False):
    """Compute the structure factor.

    The consdered trajectory must include valid elements.

    The computed structure factor is only valid for certain values of Q. The
    lowest value of Q that can sufficiently be described by a box of
    characteristic length `L` is `2 * pi / (L / 2)`.

    Parameters
    ----------
    trj : mdtraj.Trajectory
        A trajectory for which the structure factor is to be computed.   
    pair : array-like, shape=(2,), optional, default=None
        Residue name pairs to calculate partial S(Q). If default=None, the function is 
        calculating total S(Q).
    Q_range : list or np.ndarray, default=(0.5, 50)
        Minimum and maximum Values of the scattering vector, in `1/nm`, to be
        considered. 
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
    Number_scale = dict()

    Q = np.logspace(np.log10(Q_range[0]),
                    np.log10(Q_range[1]),
                    num=n_points)
    S = np.zeros(shape=(len(Q)))

    for elem in elements:
        compositions[elem.symbol] = len(top.select('element {}'.format(elem.symbol)))/trj.n_atoms
        form_factors[elem.symbol] = elem.atomic_number
        
    denom = 0
    for elem in elements:
        denom += (compositions[elem.symbol] * form_factors[elem.symbol])
    
    if pair == None:
        for i, q in enumerate(Q):
            num = 0
            for (elem1, elem2) in it.product(elements, repeat=2):
                e1 = elem1.symbol
                e2 = elem2.symbol

                f_a = form_factors[e1]
                f_b = form_factors[e2]

                x_a = compositions[e1]
                x_b = compositions[e2]

                try:
                    g_r = rdfs['{0}{1}'.format(e1, e2)]
                except KeyError:
                    pairs = top.select_pairs(selection1=top.select('element {}'.format(e1)),
                                             selection2=top.select('element {}'.format(e2)))
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
                pre_factor = x_a * x_b * f_a * f_b * 4 * np.pi * rho
                num += pre_factor * integral
            S[i] = num
    
    else:
        for i, q in enumerate(Q):
            num = 0
            for (elem1, elem2) in it.product(elements, repeat=2):
                e1 = elem1.symbol
                e2 = elem2.symbol

                f_a = form_factors[e1]
                f_b = form_factors[e2]

                x_a = compositions[e1]
                x_b = compositions[e2]

                try:
                    g_r = rdfs['{0}{1}'.format(e1, e2)]
                    scale = Number_scale['{0}{1}'.format(e1, e2)]
                except KeyError:
                    element_1 = top.select('resname {} and element {}'.format(pair[0], e1))
                    element_2 = top.select('resname {} and element {}'.format(pair[1], e2))
                    n_element_1 = len(element_1)
                    n_element_2 = len(element_2)
                    if (n_element_1 == 0) or (n_element_2 == 0):
                        g_r = np.array([0])
                        rdfs['{0}{1}'.format(e1, e2)] = g_r
                        scale = 0
                        Number_scale['{0}{1}'.format(e1, e2)] = scale
                    else:
                        pairs = top.select_pairs(selection1=element_1,
                                                 selection2=element_2)

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
                        N_i = len(top.select('element {}'.format(e1)))
                        N_j = len(top.select('element {}'.format(e2)))

                        N_i_1 = n_element_1
                        N_j_2 = n_element_2

                        scale = N_i_1 *  N_j_2 / (N_i * N_j)
                        Number_scale['{0}{1}'.format(e1, e2)] = scale
                if len(g_r) > 1:
                    integral = simps(r ** 2 * (g_r - 1) * np.sin(q * r) / (q * r), r) * scale
                else:
                    integral = 0                
                pre_factor = x_a * x_b * f_a * f_b * 4 * np.pi * rho 
                num += pre_factor * integral
            S[i] = num
    S = S/(denom**2)
    return Q, S


def structure_factor_anypair(trj, pair_1_index=None, pair_2_index=None, pair=None, Q_range=(0.5, 50), n_points=1000, framewise_rdf=False):
    """Compute the structure factor.

    The consdered trajectory must include valid elements.

    The computed structure factor is only valid for certain values of Q. The
    lowest value of Q that can sufficiently be described by a box of
    characteristic length `L` is `2 * pi / (L / 2)`.

    Parameters
    ----------
    trj : mdtraj.Trajectory
        A trajectory for which the structure factor is to be computed.  
    trj_pair: mdtraj.Trajectory
        This trajectory has renamed atom name. For example, all head atoms have the atom name, 'head'.   
    pair_1_index:
        This index corresponds to the pair[0], which is not the index of molecules but the index of subgroups of molecules.
        Note: If pair for molecule pairs (not for a group in a molecule), the pair_1_index should be None.
    pair_2_index:
        This index corresponds to the pair[1], which is not the index of molecules but the index of subgroups of molecules.
        Note: If pair for molecule pairs (not for a group in a molecule), the pair_2_index should be None.
    pair : array-like, shape=(2,), optional, default=None
        Residue name pairs to calculate partial S(Q). If default=None, the function is 
        calculating total S(Q).
    Q_range : list or np.ndarray, default=(0.5, 50)
        Minimum and maximum Values of the scattering vector, in `1/nm`, to be
        considered. 
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
    
    top = trj.topology
    if len(pair_1_index) != 0:
        mol_1_index = top.select('resname {}'.format(pair[0]))
        trj_1 = trj.atom_slice(mol_1_index)
        n_atoms = trj_1.topology.n_atoms/trj_1.topology.n_residues
        pair_1_index = find_index(pair_1_index, mol_1_index, n_atoms)
    
    if len(pair_2_index) != 0:
        mol_2_index = top.select('resname {}'.format(pair[1]))
        trj_2 = trj.atom_slice(mol_2_index)
        n_atoms = trj_2.topology.n_atoms/trj_2.topology.n_residues
        pair_2_index = find_index(pair_2_index, mol_2_index, n_atoms)
    
    rho = np.mean(trj.n_atoms / trj.unitcell_volumes)
    L = np.min(trj.unitcell_lengths)
    
    elements = set([a.element for a in top.atoms])
    compositions = dict()
    form_factors = dict()
    rdfs = dict()
    Number_scale = dict()

    Q = np.logspace(np.log10(Q_range[0]),
                    np.log10(Q_range[1]),
                    num=n_points)
    S = np.zeros(shape=(len(Q)))

    for elem in elements:
        compositions[elem.symbol] = len(top.select('element {}'.format(elem.symbol)))/trj.n_atoms
        form_factors[elem.symbol] = elem.atomic_number
        
    denom = 0
    for elem in elements:
        denom += (compositions[elem.symbol] * form_factors[elem.symbol])
    
    for i, q in enumerate(Q):
        num = 0
        for (elem1, elem2) in it.product(elements, repeat=2):
            e1 = elem1.symbol
            e2 = elem2.symbol

            f_a = form_factors[e1]
            f_b = form_factors[e2]

            x_a = compositions[e1]
            x_b = compositions[e2]

            try:
                g_r = rdfs['{0}{1}'.format(e1, e2)]
                scale = Number_scale['{0}{1}'.format(e1, e2)]
            except KeyError:
                if len(pair_1_index) == 0:
                    element_1 = top.select('resname {} and element {}'.format(pair[0], e1))
                else:
                    element_1_e1 = top.select('element {}'.format(e1))
                    element_1 = np.intersect1d(element_1_e1, pair_1_index)
                
                if len(pair_2_index) == 0:
                    element_2 = top.select('resname {} and element {}'.format(pair[1], e2))
                else:
                    element_2_e2 = top.select('element {}'.format(e2))
                    element_2 = np.intersect1d(element_2_e2, pair_2_index)
                
                n_element_1 = len(element_1)
                n_element_2 = len(element_2)
                if (n_element_1 == 0) or (n_element_2 == 0):
                    g_r = np.array([0])
                    rdfs['{0}{1}'.format(e1, e2)] = g_r
                    scale = 0
                    Number_scale['{0}{1}'.format(e1, e2)] = scale
                else:
                    pairs = top.select_pairs(selection1=element_1,
                                             selection2=element_2)

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
                    N_i = len(top.select('element {}'.format(e1)))
                    N_j = len(top.select('element {}'.format(e2)))

                    N_i_1 = n_element_1
                    N_j_2 = n_element_2

                    scale = N_i_1 *  N_j_2 / (N_i * N_j)
                    Number_scale['{0}{1}'.format(e1, e2)] = scale
            if len(g_r) > 1:
                integral = simps(r ** 2 * (g_r - 1) * np.sin(q * r) / (q * r), r) * scale
            else:
                integral = 0                
            pre_factor = x_a * x_b * f_a * f_b * 4 * np.pi * rho 
            num += pre_factor * integral
        S[i] = num
    S = S/(denom**2)
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

def find_index(initial_index, mol_index, n_atoms):
    all_index = initial_index + mol_index[0]
    times = len(mol_index)/n_atoms
    for i in range(int(times-1)):
        jump_n_atoms = all_index[-len(initial_index):] + n_atoms
        all_index = np.append(all_index, jump_n_atoms)
    return all_index