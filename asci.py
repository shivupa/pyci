import scipy as sp
import scipy as sp
import scipy.linalg as spla
import numpy as np
from functools import reduce
import pyscf
import itertools
import h5py
from pyscf import gto, scf, ao2mo, fci
'''
Energy curve by FCI
'''
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

#############
# INITIALIZE
#############
s = mol.intor('cint1e_nuc_sph')
size_basis = len(s[0])*2
print size_basis
C_new = np.zeros(sp.special.binom(size_basis,mol.nelectron))
C_old = np.zeros(sp.special.binom(size_basis,mol.nelectron))
C_old[0] = 1
C_new = np.zeros(sp.special.binom(size_basis,mol.nelectron))

myhf = scf.RHF(mol)
E = myhf.kernel()

core = np.argsort(C_old)
detlist = np.array(list(itertools.combinations(np.arange(size_basis),mol.nelectron)))

H = np.zeros((len(C_old),len(C_old)))
print "A",np.size(H)

#############
# LOOP
#############
# get fock matrix
h1e = myhf.get_hcore(mol)
s1e = myhf.get_ovlp(mol)
ijkl = ao2mo.incore.full(pyscf.scf._vhf.int2e_sph(mol._atm, mol._bas, mol._env), myhf.mo_coeff, compact=False)
mocc = myhf.mo_coeff[:,myhf.mo_occ>0]
mvir = myhf.mo_coeff[:,myhf.mo_occ==0]
ao2mo.general(mol, (mocc,mocc,mocc,mocc), 'tmp.h5', compact=False)
feri = h5py.File('tmp.h5')
gpqpq = np.array(feri['eri_mo'])
print np.shape(gpqpq)
# eq 5
for i in range(len(H)):
    occ = detlist[i]
    virt = np.setdiff1d(np.arange(size_basis),occ)
    for p in occ:
        H[i,i] += h1e[p,p]
        for q in occ:
            if p != q:
                
                print H[i,i]




enuc = mol.energy_nuc()
hcore = mol.intor('cint1e_nuc_sph') + mol.intor('cint1e_kin_sph')
ovlp = mol.intor('cint1e_ovlp_sph')
eigenvalues,L = spla.eigh(ovlp)
Lambda = np.zeros((len(eigenvalues),len(eigenvalues)))
