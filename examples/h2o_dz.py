#!/usr/bin/env python

import pyscf


mol = pyscf.gto.M(
    atom=[['O', (0.000000000000, -0.143225816552, 0.000000000000)],
          ['H', (1.638036840407, 1.136548822547, -0.000000000000)],
          ['H', (-1.638036840407, 1.136548822547, -0.000000000000)]],
    #basis='6-31G',
    basis='sto-3g',
    verbose=1,
    unit='b',
    symmetry=False,
)

if __name__ == '__main__':

    from pyci.tests.test_runner import test_runner
    test_runner(mol)
