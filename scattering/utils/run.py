import numpy as np
import matplotlib.pyplot as plt
import mdtraj as md
from scattering.van_hove import compute_van_hove
from scattering.utils.utils import get_dt
from scattering.utils.features import find_local_maxima

def run_total_vhf(trj, chunk_length, n_chunks, parallel=False, water=True,
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
    for idx, start in enumerate(starting_frames):
        end = start + chunk_length
        chunk = trj[start:end]
        print(f"Analyzing frames {start} to {end}...")
        r, t, g_r_t = compute_van_hove(trj=chunk,
                                       chunk_length=chunk_length,
                                       parallel=False,
                                       water=True,
                                       partial=False)
      
        vhf_list.append(g_r_t)

    vhf_mean = np.mean(vhf_list, axis=0)
    t_save = t - t[0]

    return r, t_save, vhf_mean
