"""
test evfuncs module
"""
import os

import numpy as np

import hvc.evfuncs

class TestEvfuncs:

    def test_load_cbin(self):
        cbin = os.path.join(os.path.dirname(__file__),
                            'test_data/cbins/gy6or6/032412/'
                            'gy6or6_baseline_240312_0811.1165.cbin')
        dat, fs = hvc.evfuncs.load_cbin(cbin)
        assert type(dat) == np.ndarray
        assert dat.dtype == '>i2'  # should be big-endian 16 bit
        assert type(fs) == int
