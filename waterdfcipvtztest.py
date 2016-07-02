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

mol = gto.M(
    atom = [['O', (0.000000000000,  -0.143225816552,   0.000000000000)],
            ['H', (1.638036840407,   1.136548822547,  -0.000000000000)],
            ['H', (-1.638036840407,   1.136548822547,  -0.000000000000)]],
    basis = 'cc-pvtz',
    verbose = 1,
    unit='b'
)
Na,Nb = mol.nelec #nelec is a tuple with (N_alpha, N_beta)
nao=mol.nao_nr()
s = mol.intor('cint1e_ovlp_sph')
t = mol.intor('cint1e_kin_sph')
v = mol.intor('cint1e_nuc_sph')
h=t+v
print np.shape(h)
#############
# FUNCTIONS
#############

# INITIALIZE
#############
myhf = scf.RHF(mol)
E = myhf.kernel()
c = myhf.mo_coeff
cisolver = fci.FCI(mol, c)
print('PYSCF  E(FCI) = %.12f' % (cisolver.kernel()[0] + mol.energy_nuc()))
