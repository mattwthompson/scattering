import numpy as np
import warnings

from mdtraj.core.element import Element

# Coherent scattering lengths from the NIST website
b_mat = {
    1: {-1: -3.7390, 1: -3.7406, 2: 6.671, 3: 4.792},
    3: {-1: -1.90, 6: 2.00 - 0.261j, 7: -2.22},
    6: {-1: 6.6460, 12: 6.6511, 13: 6.19},
    7: {-1: 9.36, 14: 9.37, 15: 6.44},
    8: {-1: 5.803, 16: 5.803, 17: 5.78, 18: 5.84},
    9: {-1: 5.654, 19: 5.654},
    11: {-1: 3.63, 23: 3.63},
    12: {-1: 5.375, 24: 5.66, 25: 3.62, 26: 4.89},
    15: {-1: 5.13, 31: 5.13},
    16: {-1: 2.847, 32: 2.804, 33: 4.74, 34: 3.48, 36: 3.1},
    17: {-1: 9.5770, 35: 11.65, 37: 3.08},
    19: {-1: 3.67, 39: 3.74, 40: 3.1, 41: 2.69},
    20: {-1: 4.70, 40: 4.80, 42: 3.36, 43: -1.56, 44: 1.42, 46: 3.6, 48: 0.39},
    35: {-1: 6.795, 79: 6.80, 81: 6.79},
    53: {-1: 5.28, 127: 5.28}
}

#def get_form_factor(element_name=None, water=None):
#
#    if water:
#        return get_form_factor_water(element_name=element_name)
#
#    if element_name is not None:
#        elem = Element.getBySymbol(element_name)
#
#    warnings.warn('Estimating atomic form factor as atomic number')
#
#    form_factor = elem.atomic_number
#    return form_factor if form_factor > 0 else 1

def get_form_factor(atomic_number, isotope_ratios, probe="neutron"):

    assert (
        np.isclose(sum(v for v in isotope_ratios.values()), 1.0), 
        f"Unnormalized fractions for atomic_number {atomic_number} in get_form_factor"
    )

    if probe == "neutron":
        b = b_mat[atomic_number]
        return [b[key]*val for key, val in isotope_ratios.items()].pop()
    else:
        warnings.warn(
            "Ignoring attenuation and treating form factor as constant equal to atomic # for X-ray calculation."
        )


def get_form_factor_water(element_name=None):
    if element_name is not None:
        elem = Element.getBySymbol(element_name)
    else:
        raise ValueError('Need an element')

    if elem.atomic_number not in [1, 8]:
        raise ValueError('')

    if elem.atomic_number == 1:
        form_factor = float(1/3)
    elif elem.atomic_number == 8:
        form_factor = float(9 + 1/3)
    else:
        raise ValueError('Found an element not in water')
        form_factor = None

    return form_factor
