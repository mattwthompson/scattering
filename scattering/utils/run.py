import numpy as np
import matplotlib.pyplot as plt
import mdtraj as md
from scattering.van_hove import compute_van_hove, compute_partial_van_hove
from scattering.utils.utils import get_dt
from scattering.utils.features import find_local_maxima

def run_total_vhf(trj, chunk_length, n_chunks, step=1, parallel=False, water=True,
                     r_range=(0, 1.0), bin_width=0.005, n_bins=None,
                     self_correlation=True, periodic=True, opt=True, partial=False):
    """ Run `calculate_van_hove` for specific number of number of chunks
        and chunk_length

    Parameters
    ----------
    trj : MDTraj.Trajectory
        MDTraj trajectory
    chunk_length : int
        Number of frames in chunk
    n_chunks : int
        Number of chunks to average over
    step : int, default=1
        Step interval of frames to analyze
    parallel : bool, default=True
        Use parallel implementation with `multiprocessing`
    water : bool
        use X-ray form factors for water that account for polarization
    r_range : array-like, shape=(2,), optional, default=(0.0, 1.0)
        Minimum and maximum radii.
    bin_width : float, optional, default=0.005
        Width of the bins in nanometers.
    n_bins : int, optional, default=None
        The number of bins. If specified, this will override the `bin_width`
         parameter.
    self_correlation : bool, default=True
        Whether or not to include the self-self correlations

    Return
    ------
    r : numpy.array
        Distances
    t : numpy.array
        times
    g_r_t : numpy.array
        van hove function averaged over n_chunks
    """
    # Calculate intervals between starting points
    starting_frames = np.linspace(0, trj.n_frames-chunk_length, n_chunks, dtype=int)
    if step != 1:
        frames_in_chunk = int(chunk_length / step)
    else:
        frames_in_chunk = chunk_length
    vhf_list = list()
    for idx, start in enumerate(starting_frames):
        end = start + chunk_length
        chunk = trj[start:end:step]
        print(f"Analyzing frames {start} to {end}...")
        r, t, g_r_t = compute_van_hove(trj=chunk,
                                       chunk_length=frames_in_chunk,
                                       parallel=parallel,
                                       water=water,
                                       r_range=r_range,
                                       bin_width=bin_width,
                                       n_bins=n_bins,
                                       self_correlation=self_correlation,
                                       periodic=periodic,
                                       opt=opt,
                                       partial=partial,)
      
        vhf_list.append(g_r_t)

    vhf_mean = np.mean(vhf_list, axis=0)
    t_save = t - t[0]

    return r, t_save, vhf_mean

def run_partial_vhf(trj, chunk_length, selection1, selection2, n_chunks, water=True,
                     step=1, r_range=(0, 1.0), bin_width=0.005, n_bins=None,
                     self_correlation=True, periodic=True, opt=True):
    """ Run `calculate_van_hove` for specific number of number of chunks
        and chunk_length

    Parameters
    ----------
    trj : MDTraj.Trajectory
        MDTraj trajectory
    chunk_length : int
        Number of frames in chunk
    n_chunks : int
        Number of chunks to average over
    step : int, default=1
        Step interval of frames to analyze
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
    self_correlation : bool, default=True
        Whether or not to include the self-self correlations

    Return
    ------
    r : numpy.array
        Distances
    t : numpy.array
        times
    g_r_t : numpy.array
        van hove function averaged over n_chunks
    """
    # Calculate intervals between starting points
    starting_frames = np.linspace(0, trj.n_frames-chunk_length, n_chunks, dtype=int)
    vhf_list = list()
    if step != 1:
        frames_in_chunk = int(chunk_length / step)
    else:
        frames_in_chunk = chunk_length
    for idx, start in enumerate(starting_frames):
        end = start + chunk_length
        chunk = trj[start:end:step]
        print(f"Analyzing frames {start} to {end}...")
        r, g_r_t = compute_partial_van_hove(trj=chunk,
                                       chunk_length=frames_in_chunk,
                                       selection1=selection1,
                                       selection2=selection2,
                                       r_range=r_range,
                                       bin_width=bin_width,
                                       n_bins=n_bins,
                                       self_correlation=self_correlation,
                                       periodic=periodic,
                                       opt=opt,)
      
        vhf_list.append(g_r_t)

    vhf_mean = np.mean(vhf_list, axis=0)
    t_save = trj.time[0:chunk_length:step]

    return r, t_save, vhf_mean
