import multiprocessing
import itertools as it
from os import PRIO_PGRP
import warnings
from psutil import virtual_memory
import progressbar


import numpy as np
import mdtraj as md

from scattering.utils.utils import get_dt
from scattering.utils.constants import get_form_factor


def compute_van_hove(trj, chunk_length, parallel=False, chunk_starts=None, cpu_count=None,
                     num_concurrent_paris=100000, water=False, r_range=(0, 1.0), bin_width=0.005, n_bins=None,
                     self_correlation=True, periodic=True, opt=True, partial=False):
    """Compute the partial van Hove function of a trajectory

    Parameters
    ----------
    trj : mdtraj.Trajectory
        trajectory on which to compute the Van Hove function
    chunk_length : int
        length of time between restarting averaging
    parallel : bool, default=True
        Use parallel implementation with `multiprocessing`
    water : bool
        use X-ray form factors for water that account for polarization
    num_concurrent_pairs : int, optional, default=100000
        number of atom pairs to compute at once
    r_range : array-like, shape=(2,), optional, default=(0.0, 1.0)
        Minimum and maximum radii.
    bin_width : float, optional, default=0.005
        Width of the bins in nanometers.
    n_bins : int, optional, default=None
        The number of bins. If specified, this will override the `bin_width`
         parameter.
    self_correlation : bool, default=True
        Whether or not to include the self-self correlations

    Returns
    -------
    r : numpy.ndarray
        r positions generated by histogram binning
    g_r_t : numpy.ndarray
        Van Hove function at each time and position
    """

    n_physical_atoms = len([a for a in trj.top.atoms if a.element.mass > 0])
    unique_elements = list(set([a.element for a in trj.top.atoms if a.element.mass > 0]))

    partial_dict = dict()

    for elem1, elem2 in it.combinations_with_replacement(unique_elements[::-1], 2):
        # Add a bool to check if self-correlations should be analyzed
        self_bool = self_correlation
        if elem1 != elem2 and self_correlation:
            self_bool = False
            warnings.warn(
                "Total VHF calculation: No self-correlations for {} and {}, setting `self_correlation` to `False`.".format(
                    elem1, elem2
                )
            )

        print('doing {0} and {1} ...'.format(elem1, elem2))
        r, g_r_t_partial = compute_partial_van_hove(trj=trj,
                                                    chunk_length=chunk_length,
                                                    selection1='element {}'.format(elem1.symbol),
                                                    selection2='element {}'.format(elem2.symbol),
                                                    chunk_starts=chunk_starts,
                                                    cpu_count=cpu_count,
                                                    num_concurrent_pairs=num_concurrent_paris,
                                                    r_range=r_range,
                                                    bin_width=bin_width,
                                                    n_bins=n_bins,
                                                    self_correlation=self_bool,
                                                    periodic=periodic,
                                                    opt=opt,
                                                    parallel=parallel,
                                                    )

        partial_dict[('element {}'.format(elem1.symbol), 'element {}'.format(elem2.symbol))] = g_r_t_partial

    if partial:
        return partial_dict

    norm = 0
    g_r_t = None

    for key, val in partial_dict.items():
        elem1, elem2 = key
        concentration1 = trj.atom_slice(trj.top.select(elem1)).n_atoms / n_physical_atoms
        concentration2 = trj.atom_slice(trj.top.select(elem2)).n_atoms / n_physical_atoms
        form_factor1 = get_form_factor(element_name=elem1.split()[1], water=water)
        form_factor2 = get_form_factor(element_name=elem2.split()[1], water=water)

        coeff = form_factor1 * concentration1 * form_factor2 * concentration2
        if g_r_t is None:
            g_r_t = np.zeros_like(val)
        g_r_t += val * coeff

        norm += coeff

    # Reshape g_r_t to better represent the discretization in both r and t
    g_r_t_final = np.empty(shape=(chunk_length, len(r)))
    for i in range(chunk_length):
        g_r_t_final[i, :] = np.mean(g_r_t[i::chunk_length], axis=0)

    g_r_t_final /= norm

    t = trj.time[:chunk_length]

    return r, t, g_r_t_final


def compute_partial_van_hove(trj, chunk_length=10, selection1=None, selection2=None, num_concurrent_pairs=100000,
                             chunk_starts=None, cpu_count=None, r_range=(0, 1.0), bin_width=0.005, 
                             n_bins=200, self_correlation=True, periodic=True, opt=True, parallel=True):
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
    num_concurrent_pairs : int, default=100000
        number of atom pairs to compute at once
    r_range : array-like, shape=(2,), optional, default=(0.0, 1.0)
        Minimum and maximum radii.
    bin_width : float, optional, default=0.005
        Width of the bins in nanometers.
    n_bins : int, optional, default=None
        The number of bins. If specified, this will override the `bin_width`
         parameter.
    self_correlation : bool, default=True
        Whether or not to include the self-self correlations
    parallel : bool, default=True
        Use parallel implementation with `multiprocessing`

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
    
    # Check if pair is monatomic
    # If not, do not calculate self correlations
    if selection1 != selection2 and self_correlation:
        warnings.warn(
            "Partial VHF calculation: No self-correlations for {} and {}, setting `self_correlation` to `False`.".format(
                selection1, selection2
                )
        )
        self_correlation = False
    
    # Don't need to store it, but this serves to check that dt is constant
    dt = get_dt(trj)

    cut_trj = trj.atom_slice(trj.top.select(str(selection1) + " or " + str(selection2)))
    del trj

    pairs = cut_trj.top.select_pairs(selection1=selection1, selection2=selection2)

    if chunk_starts is None:
        chunk_starts = []
        for i in range(cut_trj.n_frames//chunk_length):
            chunk_starts.append(i*chunk_length)
    
    if parallel:
        if cpu_count == None:    
            cpu_count = min(multiprocessing.cpu_count(), virtual_memory().total // 1024**3)
        result = None
        with multiprocessing.Pool(processes = cpu_count, maxtasksperchild = 1) as pool:
            
            
            
            result=[]
            output = pool.imap_unordered(_worker,_data(cut_trj, 
                                                       chunk_starts,
                                                       pairs,
                                                       chunk_length, 
                                                       num_concurrent_pairs,
                                                       r_range, 
                                                       bin_width, 
                                                       n_bins, 
                                                       self_correlation, 
                                                       periodic, 
                                                       opt, 
                                                       ))
            pool.close()
            pool.join()
            for i in output:
                result.append(i)
    else:
        result = []
        data = _data(cut_trj, 
                     chunk_starts,
                     pairs,
                     chunk_length, 
                     num_concurrent_pairs,
                     r_range, 
                     bin_width, 
                     n_bins, 
                     self_correlation, 
                     periodic, 
                     opt, 
                     )
        for i in data:
            result.append(_worker(data))
    
    r = []
    for val in result:
        r.append(val[0])
    r = np.mean(r, axis=0)
    
    g_r_t = []
    for val in result:
        g_r_t.append(val[1])
    g_r_t = np.mean(g_r_t, axis=0)
    
    return r, g_r_t

def _worker(input_list):
    trj, pairs, chunk_length, num_concurrent_pairs, r_range, bin_width, n_bins, self_correlation, periodic, opt = input_list
    
    times = list()
   
    for j in range(chunk_length):
        times.append([0,j])
    
    r, g_r_t_frame = md.compute_rdf_t(
        traj=trj,
        pairs=pairs,
        times=times,
        num_concurrent_pairs=num_concurrent_pairs,
        r_range=r_range,
        bin_width=bin_width,
        n_bins=n_bins,
        period_length=chunk_length,
        self_correlation=self_correlation,
        periodic=periodic,
        opt=opt,
    )
    return [r, g_r_t_frame]

def _data(trj, chunk_starts, pairs, chunk_length, num_concurrent_pairs, r_range, bin_width, n_bins, self_correlation, periodic, opt):
    for start in progressbar.progressbar(chunk_starts):
        yield ([ 
            trj[start:start+chunk_length], 
            pairs,
            chunk_length, 
            num_concurrent_pairs,
            r_range, 
            bin_width, 
            n_bins, 
            self_correlation, 
            periodic, 
            opt, 
        ])