import scipy as sp
import scipy as sp
import scipy.linalg as spla
import numpy as np

'''
Energy curve by FCI
'''

from functools import reduce
import numpy
from pyscf import gto, scf, ao2mo, fci

mol = gto.M(
    atom = [['O', (0.000000000000,  -0.143225816552,   0.000000000000)],
            ['H', (1.638036840407,   1.136548822547,  -0.000000000000)],
            ['H', (-1.638036840407,   1.136548822547,  -0.000000000000)]],
    basis = 'STO-3G',
    verbose = 1,
    unit='b'
)

#############
# INPUT
#############
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

for i in range(len(H)):
    H[i,i] = 
    
for i in core:
    occ = det[i]
    virt = np.set1d(np.arange(size_basis),occ)
    for p in range(mol.nelectron):
        for r in range(mol.nelectron):
            


enuc = mol.energy_nuc()
hcore = mol.intor('cint1e_nuc_sph') + mol.intor('cint1e_kin_sph')
ovlp = mol.intor('cint1e_ovlp_sph')
eigenvalues,L = spla.eigh(ovlp)
Lambda = np.zeros((len(eigenvalues),len(eigenvalues)))
