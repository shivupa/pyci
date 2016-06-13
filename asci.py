import scipy as sp
import scipy.linalg as spla
import numpy as np
from functools import reduce
import pyscf
import itertools
import h5py
from pyscf import gto, scf, ao2mo, fci
import pyscf.tools as pt
#############
# INPUT
#############
mol = gto.M(
    atom = [['O', (0.000000000000,  -0.143225816552,   0.000000000000)],
            ['H', (1.638036840407,   1.136548822547,  -0.000000000000)],
            ['H', (-1.638036840407,   1.136548822547,  -0.000000000000)]],
    basis = 'STO-3G',
    verbose = 1,
    unit='b'
)
cdets = 250
tdets = 500
#############
# FUNCTIONS
#############
def create_PYSCF_fcidump():
    myhf = scf.RHF(mol)
    E = myhf.kernel()
    c = myhf.mo_coeff
    h1e = reduce(np.dot, (c.T, myhf.get_hcore(), c))
    eri = ao2mo.kernel(mol, c)
    pt.fcidump.from_integrals('fcidump.txt', h1e, eri, c.shape[1],mol.nelectron, ms=0)
    cisolver = fci.FCI(mol, myhf.mo_coeff)
    print('E(HF) = %.12f, E(FCI) = %.12f' % (E,(cisolver.kernel()[0] + mol.energy_nuc())))
#############
# INITIALIZE
#############
#try except to open file
create_PYSCF_fcidump()
