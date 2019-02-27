import seaborn as sns
import pandas as pd
import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
from mdtraj.geometry.distance import compute_distances, compute_distances_t

trj = md.load('test/files/10fs_unwrapped.xtc', top='test/files/10fs.gro')[:1000]

pairs = trj.top.select_pairs('name O', 'name O')
period_length = 20
n_chunks = int(trj.n_frames / period_length)

times = list()
for i in range(n_chunks):
    for j in range(period_length):
        times.append([j+period_length*i, i*period_length])

r, g_r = md.compute_rdf_t(trj, period_length=period_length, times=times, pairs=pairs, r_range=(0, 0.8), periodic=True, opt=True)

df = pd.DataFrame(g_r[::-1] - 1)
sns.heatmap(df, vmin=-0.05, vmax=0.05, center=0, cmap='RdYlGn_r')
plt.savefig('tmp.pdf')

plt.figure(figsize=(4, 3))
for i in range(period_length):
    if i % 2 == 0:
        plt.plot(r, n_chunks * np.mean(g_r[i::period_length], axis=0), label=trj.time[i])
plt.legend()
plt.xlim((0, 0.8))
plt.ylim((0, 3.5))
plt.savefig('vhf.pdf')
