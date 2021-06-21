import multiprocessing
#import sys
#import itertools as it
#import time
#
import numpy as np
#import mdtraj as md
#from progressbar import ProgressBar
#
#from scattering.utils.utils import get_dt
#from scattering.utils.constants import get_form_factor
from scattering.van_hove import compute_van_hove

def compute_van_hove_para(trj, chunk_length, chunk_starts, water=False,
                     r_range=(0, 1.0), bin_width=0.005, n_bins=None,
                     self_correlation=True, periodic=True, opt=True, partial=False): 

    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes = cpu_count)
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
            water, 
            r_range, 
            bin_width, 
            n_bins, 
            self_correlation, 
            periodic, 
            opt, 
        ]) 

    print('A')
    help(pool.map)
    pool.map(worker,data,chunksize = (cpu_count//len(chunk_starts)) + 1)
    pool.close()
    pool.join()
    

#   manager = multiprocessing.Manager()
#   partial_dict = manager.dict()
#   jobs = []
#   version_info = sys.version_info
#   #current_chunk = 0
#   cpu_count = multiprocessing.cpu_count()
#   #q = Queue(cpu_count)
#   
#   for d in data:
#       #with multiprocessing.Pool(processes=cpu_count) as pool:
#       while len(multiprocessing.active_children()) > cpu_count:
#           time.sleep(0.2)
#       if version_info.major == 3 and version_info.minor <= 7:
#           p = Process(target=worker, args=(partial_dict, d))
#       elif version_info.major == 3 and version_info.minor >= 8:
#           ctx = multiprocessing.get_context()
#           p = Process(ctx, target=worker, args=(partial_dict, d))
#       np.append(jobs,p)
#       p.start()
#   
#   #for job in jobs:
#   #    while q.full():
#   #        time.sleep(0.1)
#   #    q.put('')
#   #    job.start()

#   while len(multiprocessing.active_children()) > 1:
#       time.sleep(0.2)

#   for proc in jobs:
#       proc.join()

    print('C')

    r = result_dict['r']
    del result_dict['r']

#   if partial:
#       return partial_dict

    g_r_t = None
#   num_chunks = len(chunk_starts)
    for key, val in result_dict.items():
        if g_r_t is None:
            g_r_t = np.zeros_like(val)
        g_r_t += val

    g_r_t /= num_chunks

#   # Reshape g_r_t to better represent the discretization in both r and t
#   # g_r_t_final = np.empty(shape=(chunk_length, len(r)))
#   # for i in range(chunk_length):
#   #     g_r_t_final[i, :] = np.mean(g_r_t[i::chunk_length], axis=0)
#   # 
#   # g_r_t_final /= norm

    t = trj.time[:chunk_length]

    print('D')
    return r, t, g_r_t


def worker(start, result_dict, chunk, chunk_length, water, r_range, bin_width, n_bins, self_correlation, periodic, opt):

    r, t, g_r_t = compute_van_hove(chunk, chunk_length, water, r_range, bin_width, n_bins, self_correlation, periodic, opt)
    result_dict[start] = g_r_t
    result_dict['r'] = r
    print('B')
#   key = data[0]
#   data.pop(0)
#   r, t, g_r_t = compute_van_hove(*data)
#   return_dict[key] = g_r_t
#   return_dict['r'] = r
