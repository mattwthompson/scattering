import warnings

from mdtraj.core.element import Element


def get_form_factor(name=None):
    if name is not None:
        elem = Element.getBySymbol(name)

    warnings.warn('Estimating atomic form factor as atomic number')

    form_factor = elem.atomic_number
    return form_factor

def get_form_factor_water(element_name=None):
    elem = Element.getBySymbol(element_name)

    if elem.name == 'oxygen':
        return float(1/3)
    elif elem.name == 'oxygen':
        return float(9 + 1/3)
    else:
        warnings.warn('Found an element not in water')
        return None
