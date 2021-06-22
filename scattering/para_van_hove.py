import multiprocessing
import numpy as np
from scattering.utils.constants import get_form_factor
from scattering.van_hove import compute_van_hove

def compute_van_hove_para(trj, chunk_length, chunk_starts, cpu_count=None, water=False,
                     r_range=(0, 1.0), bin_width=0.005, n_bins=None,
                     self_correlation=True, periodic=True, opt=True, partial=False): 

    if cpu_count == None:    
        cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes = cpu_count, maxtasksperchild = 1)
    manager = multiprocessing.Manager()
    result_dict = manager.dict()

    data = [] 
    for start in chunk_starts: 
        end = start + chunk_length 
        chunk = trj[start:end] 
        data.append([ 
            start, 
            result_dict,
            chunk, 
            chunk_length, 
            False,
            water, 
            r_range, 
            bin_width, 
            n_bins, 
            self_correlation, 
            periodic, 
            opt, 
        ]) 

    pool.starmap(worker,data)
    pool.close()
    pool.join()

    r = result_dict['r']
    del result_dict['r']

    g_r_t = None

    for key, val in result_dict.items():
        if g_r_t is None:
            g_r_t = np.zeros_like(val)
        g_r_t += val

    g_r_t /= len(chunk_starts)

    t = trj.time[:chunk_length]

    return r, t, g_r_t


def worker(start, result_dict, chunk, chunk_length, parallel, water, r_range, bin_width, n_bins, self_correlation, periodic, opt):

    r, t, g_r_t = compute_van_hove(chunk, chunk_length, parallel, water, r_range, bin_width, n_bins, self_correlation, periodic, opt)
    result_dict[start] = g_r_t
    result_dict['r'] = r
