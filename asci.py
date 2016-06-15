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
cdets = 25
tdets = 50
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
#############
# INITIALIZE
#############
myhf = scf.RHF(mol)
E = myhf.kernel()
c = myhf.mo_coeff
h1e = reduce(np.dot, (c.T, myhf.get_hcore(), c))
eri = ao2mo.kernel(mol, c)
print h1e
print eri
print np.shape(h1e),np.shape(eri)
print mol.nelectron, np.shape(h1e)[0]*2
num_occ = mol.nelectron
num_virt = ((np.shape(h1e)[0]*2)-mol.nelectron)
bitstring = "1"*num_occ
bitstring += "0"*num_virt
print bitstring
starting_amplitude =0.1
detdict = {bitstring:starting_amplitude}
print detdict
# a^dagger_i a_j |psi>

for det in detdict:
    occ_index = []
    virt_index = []
    count = 0
    for i in det:
        if i == "1":
            occ_index.append(count)
        else:
            virt_index.append(count)
        count +=1
    print occ_index
    print virt_index
    for i in occ_index:
        for j in virt_index:
            temp_det = list(det)
            temp_det[i] = "0"
            temp_det[j] = "1"
            print ''.join(temp_det)
