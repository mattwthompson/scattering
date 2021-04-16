import warnings

from mdtraj.core.element import Element


def get_form_factor(element_name=None, water=None):
    if element_name is None:
        raise ValueError('Need an element')

    elem = Element.getBySymbol(element_name)

    if water:
        if elem.atomic_number in [1, 8]:
            return get_form_factor_water(element_name=element_name, elem=elem)
        else: warnings.warn('`water` has been set to True but `element_name` does not equal `O` or `H`.')

    warnings.warn('Estimating atomic form factor as atomic number')

    form_factor = elem.atomic_number
    return form_factor if form_factor > 0 else 1

def get_form_factor_water(element_name=None, elem=None):
    elem = Element.getBySymbol(element_name)
    if elem.atomic_number == 1:
        form_factor = float(1/3)
    elif elem.atomic_number == 8:
        form_factor = float(9 + 1/3)
    else:
        raise ValueError('Found an element not in water')
        form_factor = None

    return form_factor
