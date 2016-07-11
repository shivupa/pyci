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
# tuning parameter
sigma = 100
# coarse graining parameter
gamma = 0.001
#convergence
convergence = 1e-10
#############
# INITIALIZE
#############
print("PYCI")
print("method: ACI")
print("convergence: ", convergence)
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
C = {hfdet:1.0}
print("\nHartree-Fock Energy: ", E_hf)
print("\nBeginning Iterations\n")
it_num = 0
while(np.abs(E_new - E_old) > convergence):
    #print("is hfdet in coreset? ", hfdet in coreset)
    it_num += 1
    E_old = E_new
    #print("Core Dets: ",len(coreset))
    #step 2
    targetdetset=set()
    for idet in coreset:
        targetdetset |= set(gen_singles_doubles(idet,nao))
    A = dict.fromkeys(targetdetset, 0.0)
    hamdict.update(populatehamdict((targetdetset | coreset),hamdict,h1e,eri))


    #equation 5

    for jdet in targetdetset:
        V = 0
        for idet in coreset:
            nexc_ij = n_excit_sets(idet,jdet)
            if nexc_ij in (1,2): # don't bother checking anything with a zero hamiltonian element
                try:
                    V += hamdict[frozenset([idet,jdet])] * C[idet]
                except:
                    V += hamdict[frozenset([jdet,idet])] * C[idet]
        delta  = (hamdict[frozenset((jdet))] - E_new)/2.0
        A[jdet] = delta - np.sqrt((delta**2.0) + V**2.0)
    #step 4
    A_sorted = sorted(list(A.items()),key=lambda i: -abs(i[1]))
    #cumulative energy error eq 6
    count = 0
    err = 0.0
    while (abs(err)<=sigma and count < len(A_sorted)):
        err += A_sorted[count][1]
        count +=1
    #step 5
    A_truncated = A_sorted[:count]
    A_dets = [i[0] for i in A_truncated]
    A_dets += [i for i in coreset]
    print("Q space size: ",len(A_truncated))
    targetham = getsmallham(A_dets,hamdict)
    eig_vals,eig_vecs = sp.sparse.linalg.eigsh(targetham,k=2*printroots)
    eig_vals_sorted = np.sort(eig_vals)[:printroots]
    E_new = eig_vals_sorted[0]
    print("Iteration {:} Energy: ".format(it_num), E_new + E_nuc)
    #step 4
    amplitudes = eig_vecs[:,np.argsort(eig_vals)[0]]
    newdet = [i for i in zip(A_dets,amplitudes)]
    C = {}
    #eq 10
    sorted_newdet = sorted(newdet,key=lambda j: -abs(j[1]))
    err = 0
    count = 0
    while abs(err) <= 1-(gamma*sigma):
        err += sorted_newdet[count][1]**2
        C[sorted_newdet[count][0]] = sorted_newdet[count][1]
        count +=1
    if sorted(newdet,key=lambda j: -abs(j[1]))[0][0] != hfdet:
        print("Biggest Contributor is NOT HF det ", sorted(newdet,key=lambda j: -abs(j[1]))[0])
    coreset = set(C.keys())
    print("")
visualize_sets(newdet,gen_dets_sets(nao,Na,Nb))

eig_vals_gamess = [-75.0129802245,
                   -74.7364625517,
                   -74.6886742417,
                   -74.6531877287]
#cisolver = fci.FCI(mol, mo_coefficients)
#efci = cisolver.kernel(nroots=printroots)[0] + mol.energy_nuc()
#print("first {:} PYCI ASCI eigvals vs PYSCF FCI eigvals vs GAMESS FCI eigvals".format(printroots))
#for i,j,k in zip(eig_vals_sorted + E_nuc, efci, eig_vals_gamess):
#    print(i,j,k)
print("first {:} pyci eigvals".format(printroots))
for i in (eig_vals_sorted + E_nuc):
    print(i)
