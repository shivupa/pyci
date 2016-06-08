import scipy as sp
import scipy as sp
import scipy.linalg as spla
import numpy as np
from functools import reduce
import pyscf
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
size_basis = len(s[0])
C_new = np.zero(sp.special.binom(size_basis,mol.nelectron))
C_old = np.zero(sp.special.binom(size_basis,mol.nelectron))
C_old[0] = 1
C_new = np.zero(sp.special.binom(size_basis,mol.nelectron))

myhf = scf.RHF(mol)
E = myhf.kernel()

core = np.argsort(C_old)
detlist = np.array(list(itertools.combinations(np.arange(size_basis),mol.nelectron)))

H = np.zeros((len(C_old),len(C_old)))

#############
# LOOP
#############
# get fock matrix
h1e = mf.get_hcore(mol)
s1e = mf.get_ovlp(mol)
ijkl = ao2mo.incore.full(pyscf.scf._vhf.int2e_sph(mol._atm, mol._bas, mol._env), myhf.mo_coeff, compact=False)
# eq 5
for i in len(H):
    occ = det[i]
    virt = np.set1d(np.arange(size_basis),occ)
    for p in occ:
        H[i,i] += h1e[p,p]
        for q in occ:
            if p != q:
                if(i>j)
                    ij = i*(i+1)/2 + j;
                else
                    ij = j*(j+1)/2 + i;
                h+=0.5*(ijkl[ij,ij]

for i in core:
    occ = det[i]
    virt = np.set1d(np.arange(size_basis),occ)
    for p in occ:
        H[i,i] =
        for r in occ:



for i in core:
    occ = det[i]
    virt = np.set1d(np.arange(size_basis),occ)
    for p in occ:
        for r in virt:




enuc = mol.energy_nuc()
hcore = mol.intor('cint1e_nuc_sph') + mol.intor('cint1e_kin_sph')
ovlp = mol.intor('cint1e_ovlp_sph')
eigenvalues,L = spla.eigh(ovlp)
Lambda = np.zeros((len(eigenvalues),len(eigenvalues)))
