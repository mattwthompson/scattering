import numpy as np
import mdtraj as md
from scattering.scattering import structure_factor, compute_dynamic_rdf
import matplotlib.pyplot as plt

trj = md.load('test/files/spce.xtc', top='test/files/spce.gro')
trj = trj.atom_slice(trj.top.select('name O'))

q, s = structure_factor(trj)

r_ij_t = compute_dynamic_rdf(trj[::100])
