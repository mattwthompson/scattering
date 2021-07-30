import warnings

from mdtraj.core.element import Element
from periodictable import cromermann


def get_form_factor(element_name=None, q=None, method="atomic", water=None):
    """Get form factor for elements"""
    if method == "cromer-mann" and q is None:
        raise ValueError("q must be a non-null value when method='cromer-mann'")

    if water:
        return get_form_factor_water(element_name=element_name)

    if element_name is not None:
        elem = Element.getBySymbol(element_name)

    if method == "atomic":
        warnings.warn('Estimating atomic form factor as atomic number')
        form_factor = elem.atomic_number
    elif method == "cromer-mann":
        form_factor = cromermann.fxrayatq(elem.symbol, q)
    else:
        raise ValueError(f"Invalid method {method}.  Use `atomic` or `cromer-mann`.")
    return form_factor if form_factor > 0 else 1

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
