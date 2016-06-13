import scipy as sp
import scipy.linalg as spla
import numpy as np
from functools import reduce
import pyscf
import itertools
import h5py
from pyscf import gto, scf, ao2mo, fci
import pyscf.tools as pt
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
# FUNCTIONS
#############
def create_PYSCF_fcidump():
    myhf = scf.RHF(mol)
    E = myhf.kernel()
    c = myhf.mo_coeff
    h1e = reduce(np.dot, (c.T, myhf.get_hcore(), c))
    eri = ao2mo.kernel(mol, c)
    pt.fcidump.from_integrals('fcidump.example1', h1e, eri, c.shape[1],mol.nelectron, ms=0)
    print "shit",h1e[0]
    print "shit", eri[0]
    print "shit", c.shape[1]
#############
# INITIALIZE
#############
#try except to open file
create_PYSCF_fcidump()

size_basis = len(mol.intor('cint1e_nuc_sph'))*2
print size_basis

C_new = np.zeros((int)(sp.special.binom(size_basis,mol.nelectron)))
C_old = np.zeros((int)(sp.special.binom(size_basis,mol.nelectron)))
C_old[0] = 1
C_new = np.zeros((int)(sp.special.binom(size_basis,mol.nelectron)))

core = np.argsort(C_old)
detlist = np.array(list(itertools.combinations(np.arange(size_basis),mol.nelectron)))

H = np.zeros((len(C_old),len(C_old)))
print "A",np.size(H)

#############
# LOOP
#############
# get fock matrix

ijkl = ao2mo.incore.full(pyscf.scf._vhf.int2e_sph(mol._atm, mol._bas, mol._env), myhf.mo_coeff, compact=False)
mocc = myhf.mo_coeff[:,myhf.mo_occ>0]
mvir = myhf.mo_coeff[:,myhf.mo_occ==0]
ao2mo.general(mol, (mocc,mocc,mocc,mocc), 'tmp.h5', compact=False)
feri = h5py.File('tmp.h5')
gpqpq = np.array(feri['eri_mo'])
print np.shape(gpqpq)
# eq 5





enuc = mol.energy_nuc()
hcore = mol.intor('cint1e_nuc_sph') + mol.intor('cint1e_kin_sph')
ovlp = mol.intor('cint1e_ovlp_sph')
eigenvalues,L = spla.eigh(ovlp)
Lambda = np.zeros((len(eigenvalues),len(eigenvalues)))
