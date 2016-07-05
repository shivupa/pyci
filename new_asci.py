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
#TODO: implement function that finds particles/holes based on set operations (will be easier with aocc,bocc lists of indices instead of docc,aocc(single),bocc(single)
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
#Number of eigenvalues desired
printroots=4
#############
# INITIALIZE
#############
Na,Nb = mol.nelec #nelec is a tuple with (N_alpha, N_beta)
nao=mol.nao_nr()
myhf = scf.RHF(mol)
E = myhf.kernel()
mo_coefficients = myhf.mo_coeff
#if you change the sign of these two orbitals, the hamiltonian matrix elements agree with those from GAMESS
#c.T[2]*=-1
#c.T[5]*=-1
cisolver = fci.FCI(mol, mo_coefficients)
efci = cisolver.kernel(nroots=printroots)[0] + mol.energy_nuc()
h1e = reduce(np.dot, (mo_coefficients.T, myhf.get_hcore(), mo_coefficients))
eri = ao2mo.kernel(mol, mo_coefficients)
#use eri[idx2(i,j),idx2(k,l)] to get (ij|kl) chemists' notation 2e- ints
#make full 4-index eris in MO basis (only for testing idx2)
num_orbs=2*nao
num_occ = mol.nelectron
num_virt = num_orbs - num_occ
fulldetlist_sets=gen_dets_sets(nao,Na,Nb)
ndets=len(fulldetlist_sets)
coredetlist_sets=gen_dets_sets_truncated(nao,Na,Nb)
cdets=len(core_sets)
#lists for csr sparse storage of hamiltonian
#if this is just for storage (and not diagonalization) then we can use a dict instead (or store as upper half of sparse matrix)
hrow=[]
hcol=[]
hval=[]
cdets = 100
tdets = 50
E_old = 0
convergence = 1e-6
C = np.zeros(cdets)
C[0] = 1
A = np.zeros(cdets)
targetdetlist_sets = []
while(E - E_old > convergence):
    #step 0
    for i in range(cdets):
        idet=coredetlist_sets[i]
        hii = calc_hii_sets(idet,h1e,eri)
        hrow.append(i)
        hcol.append(i)
        hval.append(hii)
        for j in range(i+1,cdets):
            jdet=coredetlist_sets[j]
            nexc_ij = n_excit_sets(idet,jdet)
            if nexc_ij in (1,2):
                if nexc_ij==1:
                    hij = calc_hij_single_sets(idet,jdet,h1e,eri)
                else:
                    hij = calc_hij_double_sets(idet,jdet,h1e,eri)
                hrow.append(i)
                hrow.append(j)
                hcol.append(j)
                hcol.append(i)
                hval.append(hij)
                hval.append(hij)
    core_ham=sp.sparse.csr_matrix((hval,(hrow,hcol)),shape=(cdets,cdets))
    #step 1
    for i in range(cdets):
        for j in range(i+1,cdets):
            A[i] += H[i,j]*C[j]
        A[i] /= coreham[i,i] - E
    #step 2
    for i in np.argsort(np.abs(A))[::-1][0:tdets]:
        targetdetlist_sets.append(coredetlist_sets[i])
    #step 3
    eig_vals,eig_vecs = sp.sparse.linalg.eigsh(fullham,k=2*printroots)
    eig_vals_sorted = sorted(eig_vals)[:printroots] + mol.energy_nuc()
eig_vals_gamess = [-75.0129802245,
                   -74.7364625517,
                   -74.6886742417,
                   -74.6531877287]
print("first {:} pyci eigvals vs PYSCF eigvals vs GAMESS eigvals".format(printroots))
for i,j,k in zip(eig_vals_sorted, efci, eig_vals_gamess):
    print(i,j,k)
