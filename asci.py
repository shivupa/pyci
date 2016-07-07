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
# size of core space
cdets = 500
# size of target space
tdets = 600
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
fulldetlist_sets=gen_dets_sets(nao,Na,Nb)
ndets=len(fulldetlist_sets)
full_hamiltonian = construct_hamiltonian(ndets,fulldetlist_sets,h1e,eri)
hamdict = construct_ham_dict(fulldetlist_sets,h1e,eri)
E_old = 0.0
E_new = E_hf
convergence = 1e-3
hfdet = (frozenset(range(Na)),frozenset(range(Nb)))
targetdetset = set()
coreset = {hfdet}
C = {hfdet:1.0}
print("Hartree-Fock Energy: ", E_hf)
print("")
it_num = 0
while(np.abs(E_new - E_old) > convergence):
    print("is hfdet in coreset? ", hfdet in coreset)
    it_num += 1
    E_old = E_new
    print("Core Dets: ",len(coreset))
    #step 1
    targetdetset=set()
    for idet in coreset:
        targetdetset |= set(gen_singles_doubles(idet,nao))
    A = dict.fromkeys(targetdetset, 0.0)
    for idet in coreset:
        for jdet in gen_singles_doubles(idet,nao):
            A[jdet] += hamdict[frozenset([idet,jdet])] * C[idet]
    for idet in targetdetset:
        A[idet] /= (hamdict[frozenset((idet))] - E_old)
    for idet in coreset:
        if idet in A:
            if abs(A[idet]) < abs(C[idet]): # replace with the biggest again
                A[idet] = C[idet]
        else:
            A[idet] = C[idet]
    A_sorted = sorted(list(A.items()),key=lambda i: -abs(i[1]))
    A_truncated = A_sorted[:tdets]
    print("Target Dets: ",len(A_truncated))
    A_dets = [i[0] for i in A_truncated]
    targetham = getsmallham(A_dets,hamdict)
    eig_vals,eig_vecs = sp.sparse.linalg.eigsh(targetham,k=2*printroots)
    eig_vals_sorted = np.sort(eig_vals)[:printroots]
    E_new = eig_vals_sorted[0]
    print("Iteration {:} Energy: ".format(it_num), E_new + E_nuc)
    #step 4
    amplitudes = eig_vecs[:,np.argsort(eig_vals)[0]]
    newdet = [i for i in zip(A_dets,amplitudes)]
    C = {}
    for i in sorted(newdet,key=lambda j: -abs(j[1]))[:cdets]:
        C[i[0]] = i[1]
    if sorted(newdet,key=lambda j: -abs(j[1]))[0][0] != hfdet:
        print("Biggest Contributor is NOT HF det ", sorted(newdet,key=lambda j: -abs(j[1]))[0])
    coreset = set(C.keys())
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
