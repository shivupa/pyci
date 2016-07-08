#!/bin/python3
import scipy as sp
import scipy.linalg as spla
import scipy.sparse.linalg as splinalg
import numpy as np
from functools import reduce
import pyscf
import itertools
import h5py
from pyscf import gto, scf, ao2mo, fci
import pyscf.tools as pt
import copy
import matplotlib.pyplot as plt
from utils import *
import time
#############
# INPUT
#############
#TODO: implement function that finds particles/holes based on set operations (will be easier with aocc,bocc lists of indices instead of docc,aocc(single),bocc(single)
np.set_printoptions(precision=4,suppress=True)
# Molecule Definition
mol = gto.M(
    atom = [['O', (0.000000000000,  -0.143225816552,   0.000000000000)],
            ['H', (1.638036840407,   1.136548822547,  -0.000000000000)],
            ['H', (-1.638036840407,   1.136548822547,  -0.000000000000)]],
    basis = '6-31G',
    #basis = 'sto-6g',
    verbose = 1,
    unit='b',
    symmetry=True
)
start = time.time()
asci(mol,50,100,10e-9,iter_min=10)
end = time.time()
print("RUN TIME ASCI: (h:m:s)")
m, s = divmod(end-start, 60)
h, m = divmod(m, 60)
print("%d:%02d:%02d" % (h, m, s))
"""
print("FCI from PYSCF")
start = time.time()
myhf = scf.RHF(mol)
E = myhf.kernel()
c = myhf.mo_coeff
cisolver = fci.FCI(mol, c)
efci = cisolver.kernel(nroots=4)[0] + mol.energy_nuc()
print(efci)
end = time.time()
print("RUN TIME PYSCF: (h:m:s)")
m, s = divmod(end-start, 60)
h, m = divmod(m, 60)
print("%d:%02d:%02d" % (h, m, s))
"""
