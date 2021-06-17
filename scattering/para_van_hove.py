import multiprocessing
import sys
import itertools as it

import numpy as np
import mdtraj as md
from progressbar import ProgressBar

from scattering.utils.utils import get_dt
from scattering.utils.constants import get_form_factor


def compute_van_hove_para(trj, chunk_length, chunk_starts, water=False,
                     r_range=(0, 1.0), bin_width=0.005, n_bins=None,
                     self_correlation=True, periodic=True, opt=True, partial=False): 

    data = []
    for start in chunk_starts:
        end = frame + chunk_length
        if end > trj.n_frames:
            continue
        chunk = trj[start:end]
        data.append([
            start,
            chunk,
            chunk_length,
            water,
            r_range,
            bin_width,
            n_bins,
            self_correlation,
            periodic,
            opt,
        ])

    manager = multiprocessing.Manager()
    partial_dict = manager.dict()
    jobs = []
    version_info = sys.version_info
    for d in data:
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            if version_info.major == 3 and version_info.minor <= 7:
                p = pool.Process(target=worker, args=(partial_dict, d))
            elif version_info.major == 3 and version_info.minor >= 8:
                ctx = multiprocessing.get_context()
                p = pool.Process(ctx, target=worker, args=(partial_dict, d))
            jobs.append(p)
            p.start()

    for proc in jobs:
            proc.join()

    r = partial_dict['r']
    del partial_dict['r']

    if partial:
        return partial_dict

    g_r_t = None

    for key, val in partial_dict.items():
        if g_r_t is None:
            g_r_t = np.zeros_like(val)
        g_r_t += val 
    
    g_r_t /= len(chunk_starts)

    # Reshape g_r_t to better represent the discretization in both r and t
    # g_r_t_final = np.empty(shape=(chunk_length, len(r)))
    # for i in range(chunk_length):
    #     g_r_t_final[i, :] = np.mean(g_r_t[i::chunk_length], axis=0)
    # 
    # g_r_t_final /= norm

    t = trj.time[:chunk_length]

    return r, t, g_r_t_final


def worker(return_dict, data):
    key = data[0]
    data.popleft()
    r, g_r_t = compute_van_hove(*data)
    return_dict[key] = g_r_t
    return_dict['r'] = r
