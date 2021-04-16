import numpy as np

from scattering.utils.constants import get_form_factor
from mdtraj.core.element import Element

def test_form_factor():
    li_form =  get_form_factor('li', water=False)

    assert li_form == 3

def test_water_form_factor():
    o_form =  get_form_factor('O', water=True)
    h_form =  get_form_factor('H', water=True)

    assert np.isclose(o_form, 9.33333)
    assert np.isclose(h_form, 1/3)

def test_redirect_water_form_factor():
    li_form =  get_form_factor('li', water=True)

    assert li_form == 3
