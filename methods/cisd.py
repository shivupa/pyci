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
#############
# INPUT
#############
np.set_printoptions(precision=4,suppress=True)
# Molecule Definition
mol = gto.M(
    atom = [['O', (0.000000000000,  -0.143225816552,   0.000000000000)],
            ['H', (1.638036840407,   1.136548822547,  -0.000000000000)],
            ['H', (-1.638036840407,   1.136548822547,  -0.000000000000)]],
    basis = 'STO-3G',
    verbose = 1,
    unit='b',
    symmetry=True
)
# Number of eigenvalues desired
printroots=4
#############
# INITIALIZE
#############
print("PYCI")
print("method: CISD")
print("Number of eigenvalues: ",printroots)
print("")
Na,Nb = mol.nelec #nelec is a tuple with (N_alpha, N_beta)
E_nuc = mol.energy_nuc()
nao = mol.nao_nr()
myhf = scf.RHF(mol)
E_hf = myhf.kernel()
mo_coefficients = myhf.mo_coeff
h1e = reduce(np.dot, (mo_coefficients.T, myhf.get_hcore(), mo_coefficients))
print("transforming eris")
eri = ao2mo.kernel(mol, mo_coefficients)
hamdict = dict()
E_old = 0.0
E_new = E_hf
hfdet = (frozenset(range(Na)),frozenset(range(Nb)))
targetdetset = set()
coreset = {hfdet}
print("\nHartree-Fock Energy: ", E_hf)
targetdetset=set()
for idet in coreset:
    targetdetset |= set(gen_singles_doubles(idet,nao))
targetdetset |= coreset
hamdict.update(populatehamdict(targetdetset,hamdict,h1e,eri))
targetham = getsmallham(list(targetdetset),hamdict)
eig_vals,eig_vecs = sp.sparse.linalg.eigsh(targetham,k=2*printroots)
eig_vals_sorted = np.sort(eig_vals)[:printroots]
E_new = eig_vals_sorted[0]
print("")


eig_vals_gamess = [-75.0129802245,
                   -74.7364625517,
                   -74.6886742417,
                   -74.6531877287]
cisolver = fci.FCI(mol, mo_coefficients)
efci = cisolver.kernel(nroots=printroots)[0] + mol.energy_nuc()
print("first {:} PYCI ASCI eigvals vs PYSCF FCI eigvals vs GAMESS FCI eigvals".format(printroots))
for i,j,k in zip(eig_vals_sorted + E_nuc, efci, eig_vals_gamess):
    print(i,j,k)
