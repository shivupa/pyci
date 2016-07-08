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
# cutoff parameter
epsilon = 0.0001
convergence = 0.01
#############
# INITIALIZE
#############
Na,Nb = mol.nelec #nelec is a tuple with (N_alpha, N_beta)
E_nuc = mol.energy_nuc()
nao = mol.nao_nr()
myhf = scf.RHF(mol)
E_hf = myhf.kernel()
mo_coefficients = myhf.mo_coeff
h1e = reduce(np.dot, (mo_coefficients.T, myhf.get_hcore(), mo_coefficients))
eri = ao2mo.kernel(mol, mo_coefficients)
num_orbs=2*nao
num_occ = mol.nelectron
num_virt = num_orbs - num_occ

E_old = 0.0
E_new = E_hf

hfdet = (frozenset(range(Na)),frozenset(range(Nb)))
oldselecteddetset= {hfdet}
C = {hfdet:1.0}
print("Hartree-Fock Energy: ", E_hf)
print("")
it_num = 1
Converged = False
hamdict=dict()
while( not Converged):
    newselecteddetset=set()
    newselecteddetset,hamdict_additions = heatbath(oldselecteddetset,nao,hamdict,C,epsilon,h1e,eri)
    hamdict.update(hamdict_additions)
    newselecteddetset |= oldselecteddetset
    hamdict.update(populatehamdict((newselecteddetset),hamdict,h1e,eri))
    selectedham = getsmallham(list(newselecteddetset),hamdict)
    eig_vals,eig_vecs = sp.sparse.linalg.eigsh(selectedham,k=2*printroots)
    eig_vals_sorted = np.sort(eig_vals)[:printroots]
    E_new = eig_vals_sorted[0]
    amplitudes = eig_vecs[:,np.argsort(eig_vals)[0]]
    newdet = [i for i in zip(newselecteddetset,amplitudes)]
    C = {}
    for i in sorted(newdet,key=lambda j: -abs(j[1])):
        C[i[0]] = i[1]
    print("Iteration {:} Energy: ".format(it_num), E_new + E_nuc)
    amount_added=abs(len(newselecteddetset)-len(oldselecteddetset))
    if amount_added < np.ceil(convergence * len(oldselecteddetset)):
        Converged = True
    else:
        print("Iteration {:} space growth %: ".format(it_num), amount_added)
    oldselecteddetset |= newselecteddetset
    it_num += 1
    E_old = E_new
    print("Selected Space size: ",len(oldselecteddetset))
    print("")


eig_vals_gamess = [-75.0129802245,
                   -74.7364625517,
                   -74.6886742417,
                   -74.6531877287]
cisolver = fci.FCI(mol, mo_coefficients)
efci = cisolver.kernel(nroots=printroots)[0] + mol.energy_nuc()
print("first {:} PYCI HBCI eigvals vs PYSCF FCI eigvals vs GAMESS FCI eigvals".format(printroots))
for i,j,k in zip(eig_vals_sorted + E_nuc, efci, eig_vals_gamess):
    print(i,j,k)
