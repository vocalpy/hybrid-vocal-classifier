"""
test evfuncs module
"""
import os

import numpy as np

import hvc.evfuncs


class TestEvfuncs:
    def test_load_cbin(self, test_data_dir):
        cbin = os.path.join(
            test_data_dir,
            os.path.normpath(
                "cbins/gy6or6/032412/" "gy6or6_baseline_240312_0811.1165.cbin"
            ),
        )
        dat, fs = hvc.evfuncs.load_cbin(cbin)
        assert type(dat) == np.ndarray
        assert dat.dtype == ">i2"  # should be big-endian 16 bit
        assert type(fs) == int
