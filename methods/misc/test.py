import scipy as sp
import scipy.linalg as spla
import numpy as np
from functools import reduce
import pyscf
import itertools
import h5py
from pyscf import gto, scf, ao2mo, fci
import pyscf.tools as pt
import copy
#############
# INPUT
#############
#2-index transformation for accessing eri elements with standard 4 indices
__idx2_cache = {}
def idx2(i,j):
    if (i,j) in __idx2_cache:
        return __idx2_cache[i,j]
    elif i>j:
        __idx2_cache[i,j] = i*(i+1)/2+j
    else:
        __idx2_cache[i,j] = j*(j+1)/2+i
    return __idx2_cache[i,j]

mol = gto.M(
    atom = [['O', (0.000000000000,  -0.143225816552,   0.000000000000)],
            ['H', (1.638036840407,   1.136548822547,  -0.000000000000)],
            ['H', (-1.638036840407,   1.136548822547,  -0.000000000000)]],
    basis = 'sto-3g',
    verbose = 1,
    unit='b'
)
cdets = 25
tdets = 50
nao=mol.nao_nr()
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
def amplitude(det,excitation):
    return 0.1
#############
# INITIALIZE
#############
s = mol.intor('cint1e_ovlp_sph')
t = mol.intor('cint1e_kin_sph')
v = mol.intor('cint1e_nuc_sph')
h=t+v
s2 = mol.intor('cint1e_ovlp_cart')
t2 = mol.intor('cint1e_kin_cart')
v2 = mol.intor('cint1e_nuc_cart')

# The one-electron part of the Hamiltonian is
# the sum of the kinetic and nuclear integrals
h2 = t2 + v2
myhf = scf.RHF(mol)
E = myhf.kernel()
hcore = myhf.get_hcore()
j = myhf.get_j()
k = myhf.get_k()
fock = hcore + j - 0.5 * k
c = myhf.mo_coeff
h1e = reduce(np.dot, (c.T, hcore, c))
fock2 = reduce(np.dot, (c.T, fock, c))
nocc=mol.nelectron/2
#rdm1 = 2*np.dot(c[:,:nocc],c[:,:nocc].T)
rdm1 = reduce(np.dot, (c,np.diag(myhf.mo_occ),c.T))
eri = ao2mo.kernel(mol, c)


#use eri[idx2(i,j),idx2(k,l)] to get (ij|kl) chemists' notation 2e- ints

#make full 4-index eris in MO basis (only for testing idx2)
eri_mo = ao2mo.restore(1, eri, nao)

#eri in AO basis

eri_ao = mol.intor('cint2e_sph',aosym=4)
eri_ao2 = mol.intor('cint2e_sph')
eri_ao4 = eri_ao2.reshape([nao,nao,nao,nao])
#one more way to get eris in MO basis
eri_mo2 = ao2mo.incore.full(eri_ao,c)
#print h1e
#print eri
#print np.shape(h1e),np.shape(eri)
#print mol.nelectron, np.shape(h1e)[0]*2
num_orbs=2*len(h1e)
num_occ = mol.nelectron
num_virt = ((np.shape(h1e)[0]*2)-mol.nelectron)
