#!/usr/bin/env python

import pyscf

from pyci.tests.test_runner import test_runner


r = 1.1941
mol = pyscf.gto.M(
    atom=[['C', (0.0, 0.0, 0.0)],
          ['N', (0.0, 0.0, r)]],
    basis='sto-3g',
    spin=1,
    verbose=1,
    symmetry=False,
)

test_runner(mol)
