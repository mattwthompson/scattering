import multiprocessing
import warnings
from psutil import virtual_memory
from progressbar import ProgressBar
from itertools import combinations_with_replacement
import numpy as np
import mdtraj as md


from scattering.utils.utils import get_dt, get_unique_atoms
from scattering.utils.constants import get_form_factor


def compute_van_hove(
    trj,
    chunk_length,
    parallel=True,
    chunk_starts=None,
    cpu_count=None,
    water=False,
    r_range=(0, 1.0),
    bin_width=0.005,
    n_bins=None,
    self_correlation=True,
    periodic=True,
    n_concurrent_pairs=100000,
    opt=True,
    partial=False,
):
    """Compute the  Van Hove function of a trajectory. Atom pairs
    referenced in partial Van Hove functions are in alphabetical
    order. If specific ordering of atom pairs are needed, user should
    use compute_partial_van_hove then vhf_from_pvhf to compute total
    Van Hove function.


    Parameters
    ----------
    trj : mdtraj.Trajectory
        trajectory on which to compute the Van Hove function
    chunk_length : int
        length of time between restarting averaging
    parallel : bool, default=True
        Use parallel implementation with `multiprocessing`
    chunk_starts : array-like, shape=(n_chunks,), optional, default=[chunk_length * i for i in range(trj.n_frames//chunk_length)]
        The first frame of each chunk to be analyzed.
    cpu_count : int, optional, default=min(multiprocessing.cpu_count(), total system memory in GB)
        The number of cpu process to run at once if parallel is True
    water : bool, optional, default=False
        Use X-ray form factors for water that account for polarization
    r_range : array-like, shape=(2,), optional, default=(0.0, 1.0)
        Minimum and maximum radii.
    bin_width : float, optional, default=0.005
        Width of the bins in nanometers.
    n_bins : int, optional, default=None
        The number of bins. If specified, this will override the `bin_width`
         parameter.
    self_correlation : bool or str, default=True, other: False, 'self'
        Whether or not to include the self-self correlations.
        If 'self', only self-correlations are computed.
    periodic : bool, optional, default=True
        Whether or not to use periodic boundary conditions
    opt : bool, optional, default=True
        Use an optimized native library to compute the pair wise distances.
    n_concurrent_pairs : int, optional, default=100000
        number of atom pairs to compute at once
    partial : bool, default = False
        Whether or not to return a dictionary including partial Van Hove function.

    Returns
    -------
    r : numpy.ndarray
        r positions generated by histogram binning
    g_r_t : numpy.ndarray
        Van Hove function at each time and position
    """
    if chunk_starts is None:
        chunk_starts = [chunk_length * i for i in range(trj.n_frames // chunk_length)]

    if chunk_starts[-1] + chunk_length > trj.n_frames:
        raise IndexError(
            "A chunk of length {} at time {} would fall beyond the end of the given trajectory".format(
                chunk_length, chunk_starts[-1]
            )
        )

    n_physical_atoms = len([a for a in trj.top.atoms if a.element.mass > 0])

    unique_elements = list(
        set([a.element for a in trj.top.atoms if a.element.mass > 0])
    )

    partial_dict = dict()

    for elem1, elem2 in combinations_with_replacement(unique_elements[::-1], 2):
        if elem1.symbol > elem2.symbol:
            temp = elem1
            elem1 = elem2
            elem2 = temp

        # Add a bool to check if self-correlations should be analyzed
        self_bool = self_correlation
        if elem1 != elem2 and self_correlation:
            self_bool = False
            warnings.warn(
                "Total VHF calculation: No self-correlations for {} and {}, setting `self_correlation` to `False`.".format(
                    elem1, elem2
                )
            )

        print("doing {0} and {1} ...".format(elem1, elem2))
        r, g_r_t_partial = compute_partial_van_hove(
            trj=trj,
            chunk_length=chunk_length,
            selection1="element {}".format(elem1.symbol),
            selection2="element {}".format(elem2.symbol),
            chunk_starts=chunk_starts,
            cpu_count=cpu_count,
            r_range=r_range,
            bin_width=bin_width,
            n_bins=n_bins,
            self_correlation=self_bool,
            periodic=periodic,
            n_concurrent_pairs=n_concurrent_pairs,
            opt=opt,
            parallel=parallel,
        )

        partial_dict[
            ("element {}".format(elem1.symbol), "element {}".format(elem2.symbol))
        ] = g_r_t_partial

    if partial:
        return partial_dict

    norm = 0
    g_r_t = None

    for key, val in partial_dict.items():
        elem1, elem2 = key
        concentration1 = (
            trj.atom_slice(trj.top.select(elem1)).n_atoms / n_physical_atoms
        )
        concentration2 = (
            trj.atom_slice(trj.top.select(elem2)).n_atoms / n_physical_atoms
        )
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


def compute_partial_van_hove(
    trj,
    chunk_length=10,
    selection1=None,
    selection2=None,
    chunk_starts=None,
    cpu_count=None,
    r_range=(0, 1.0),
    bin_width=0.005,
    n_bins=200,
    self_correlation=True,
    periodic=True,
    n_concurrent_pairs=100000,
    opt=True,
    parallel=True,
):

    """Compute the partial van Hove function of a trajectory

    Parameters
    ----------
    trj : mdtraj.Trajectory
        trajectory on which to compute the Van Hove function
    chunk_length : int, default=10
        length of time between restarting averaging
    selection1 : str
        selection to be considered, in the style of MDTraj atom selection
    selection2 : str
        selection to be considered, in the style of MDTraj atom selection
    chunk_starts : int array-like, shape=(n_chunks,), optional, default=[chunk_length * i for i in range(trj.n_frames//chunk_length)]
        The first frame of each chunk to be analyzed.
    cpu_count : int, optional, default=min(multiprocessing.cpu_count(), total system memory in GB)
        The number of cpu processes to simultaneously run if parallel=True
    r_range : array-like, shape=(2,), optional, default=(0.0, 1.0)
        Minimum and maximum radii.
    bin_width : float, optional, default=0.005
        Width of the bins in nanometers.
    n_bins : int, optional, default=None
        The number of bins. If specified, this will override the `bin_width`
         parameter.qq
    self_correlation : bool or str, default=True, other: False, 'self'
        Whether or not to include the self-self correlations.
        if 'self', only self-correlations are computed.
    periodic : bool, optional, default=True
        Whether or not to use periodic boundary conditions
    n_concurrent_pairs : int, default=100000
        number of atom pairs to compute at once
    opt : bool, optional, default=True
        Use an optimized native library to compute the pair wise distances.
    parallel : bool, default=True
        Use parallel implementation with `multiprocessing`

    Returns
    -------
    r : numpy.ndarray
        r positions generated by histogram binning
    g_r_t : numpy.ndarray
        Van Hove function at each time and position
    """

    if chunk_starts is None:
        chunk_starts = [chunk_length * i for i in range(trj.n_frames // chunk_length)]

    for i in chunk_starts:
        if i + chunk_length > trj.n_frames:
            raise IndexError(
                "A chunk of length {} at time {} would fall beyond the end of the given trajectory".format(
                    chunk_length, i
                )
            )

    unique_elements = (
        set([a.element for a in trj.atom_slice(trj.top.select(selection1)).top.atoms]),
        set([a.element for a in trj.atom_slice(trj.top.select(selection2)).top.atoms]),
    )

    if any([len(val) > 1 for val in unique_elements]):
        raise UserWarning(
            "Multiple elements found in a selection(s). Results may not be "
            "direcitly comprable to scattering experiments."
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
    # Question, could we just call get_dt(~) here instead of setting the variable dt = get_dt(~)
    dt = get_dt(trj)

    if parallel:
        if cpu_count == None:
            cpu_count = min(
                multiprocessing.cpu_count(), virtual_memory().total // 1024 ** 3
            )
        result = []
        with multiprocessing.Pool(processes=cpu_count, maxtasksperchild=1) as pool:
            pBar = ProgressBar(max_value=len(chunk_starts))
            for output in pBar(
                pool.imap_unordered(
                    _worker,
                    _data(
                        trj,
                        chunk_starts,
                        selection1,
                        selection2,
                        chunk_length,
                        r_range,
                        bin_width,
                        n_bins,
                        self_correlation,
                        periodic,
                        n_concurrent_pairs,
                        opt,
                    ),
                )
            ):
                result.append(output)
            pool.terminate()
            pool.join()
    else:
        result = []
        data = _data(
            trj,
            chunk_starts,
            selection1,
            selection2,
            chunk_length,
            r_range,
            bin_width,
            n_bins,
            self_correlation,
            periodic,
            n_concurrent_pairs,
            opt,
        )
        pBar = ProgressBar(max_value=len(chunk_starts))
        for i in pBar(data, max_value=len(chunk_starts)):
            result.append(_worker(i))

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

    (
        trj,
        pairs,
        chunk_length,
        r_range,
        bin_width,
        n_bins,
        self_correlation,
        periodic,
        n_concurrent_pairs,
        opt,
    ) = input_list

    times = list()

    for j in range(chunk_length):
        times.append([0, j])

    r, g_r_t_frame = md.compute_rdf_t(
        traj=trj,
        pairs=pairs,
        times=times,
        r_range=r_range,
        bin_width=bin_width,
        n_bins=n_bins,
        period_length=chunk_length,
        self_correlation=self_correlation,
        periodic=periodic,
        n_concurrent_pairs=n_concurrent_pairs,
        opt=opt,
    )
    return [r, g_r_t_frame]


def _data(
    trj,
    chunk_starts,
    selection1,
    selection2,
    chunk_length,
    r_range,
    bin_width,
    n_bins,
    self_correlation,
    periodic,
    n_concurrent_pairs,
    opt,
):
    for start in chunk_starts:
        short_trj = trj[start : start + chunk_length]
        short_trj = short_trj.atom_slice(short_trj.top.select(str(selection1) + " or " + str(selection2)))
        pairs = short_trj.top.select_pairs(selection1=selection1, selection2=selection2)
        if self_correlation == 'self':
            pairs_set = np.unique(pairs)
            pairs = np.vstack([pairs_set, pairs_set]).T
            # TODO: Find better way to only use self-pairs
            # This is hacky right now
            self_correlation = False
        yield (
            [
                short_trj,
                pairs,
                chunk_length,
                r_range,
                bin_width,
                n_bins,
                self_correlation,
                periodic,
                n_concurrent_pairs,
                opt,
            ]
        )


def vhf_from_pvhf(trj, partial_dict, water=False):
    """
    Compute the total Van Hove function from partial Van Hove functions


    Parameters
    ----------
    trj : mdtrj.Trajectory
        trajectory on which partial vhf were calculated form
    partial_dict : dict
        dictionary containing partial vhf as a np.array.
        Key is a tuple of len 2 with 2 atom types

    Return
    -------
    total_grt : numpy.ndarray
        Total Van Hove Function generated from addition of partial Van Hove Functions
    """
    unique_atoms = get_unique_atoms(trj)
    all_atoms = [atom for atom in trj.topology.atoms]

    norm_coeff = 0
    dict_shape = list(partial_dict.values())[0][0].shape
    total_grt = np.zeros(dict_shape)

    for atom_pair in partial_dict.keys():
        # checks if key is a tuple
        if isinstance(atom_pair, tuple) == False:
            raise ValueError("Dictionary key not valid. Must be a tuple.")
        for atom in atom_pair:
            # checks if the atoms in tuple pair are atom types
            if type(atom) != type(unique_atoms[0]):
                raise ValueError(
                    "Dictionary key not valid. Must be type `MDTraj.Atom`."
                )
            # checks if atoms are in the trajectory
            if atom not in all_atoms:
                raise ValueError(
                    f"Dictionary key not valid, `Atom` {atom} not in MDTraj trajectory."
                )

        # checks if key has two atoms
        if len(atom_pair) != 2:
            raise ValueError(
                "Dictionary key not valid. Must only have 2 atoms per pair."
            )

        atom1 = atom_pair[0]
        atom2 = atom_pair[1]
        coeff = (
            get_form_factor(element_name=f"{atom1.element.symbol}", water=False)
            * get_form_factor(element_name=f"{atom2.element.symbol}", water=False)
            * len(trj.topology.select(f"name {atom1.name}"))
            / (trj.n_atoms)
            * len(trj.topology.select(f"name {atom2.name}"))
            / (trj.n_atoms)
        )

        normalized_pvhf = coeff * partial_dict[atom_pair]
        norm_coeff += coeff
        total_grt = np.add(total_grt, normalized_pvhf)

    total_grt /= norm_coeff

    return total_grt