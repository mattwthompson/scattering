import warnings

from mdtraj.core.element import Element


def get_form_factor(name=None):
    if name is not None:
        elem = Element.getBySymbol(name)

    warnings.warn('Estimating atomic form factor as atomic number')

    form_factor = elem.atomic_number
    return form_factor
