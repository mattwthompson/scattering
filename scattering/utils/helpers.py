import numpy as np
import mdtraj as md

def rdf_by_frame(trj, **kwargs):
    """Helper function that computes rdf frame-wise and returns the average
    rdf over all frames. Can be useful for large systems in which a
    distance array of size n_frames * n_atoms (used internally in
    md.compute_rdf) takes requires too much memory.
    """
    g_r = None
    for frame in trj:
        r, g_r_frame = md.compute_rdf(frame, **kwargs) 
        if g_r_all is None:
            g_r_all = np.zeros_like(g_r)
        g_r += g_r_frame
    g_r /= len(trj)
    return r, g_r
