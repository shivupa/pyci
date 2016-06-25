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
def amplitude(det,excitation):

    return 0.1
#############
# INITIALIZE
#############
myhf = scf.RHF(mol)
E = myhf.kernel()
c = myhf.mo_coeff
h1e = reduce(np.dot, (c.T, myhf.get_hcore(), c))
eri = ao2mo.kernel(mol, c)
#print h1e
#print eri
#print np.shape(h1e),np.shape(eri)
#print mol.nelectron, np.shape(h1e)[0]*2
num_occ = mol.nelectron
num_virt = ((np.shape(h1e)[0]*2)-mol.nelectron)
bitstring = "1"*num_occ
bitstring += "0"*num_virt
print bitstring
starting_amplitude =1.0
original_detdict = {bitstring:starting_amplitude}

H_core = np.array((cdets,cdets))
H_target = np.array((tdets,tdets))
#############
# MAIN LOOP
#############
# a^dagger_i a_j |psi>
temp_detdict = copy.deepcopy(original_detdict)
temp_double_detdict = copy.deepcopy(original_detdict)
print temp_detdict
for det in original_detdict:
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
            temp_det =  ''.join(temp_det)
            temp_detdict[temp_det] = amplitude(det,temp_det)
            #print temp_det, temp_amplitude
            for k in occ_index:
                for l in virt_index:
                    if k>i and l>j:
                        temp_double_det = list(det)
                        temp_double_det[i] = "0"
                        temp_double_det[j] = "1"
                        temp_double_det[k] = "0"
                        temp_double_det[l] = "1"
                        temp_double_det =  ''.join(temp_det)
                        temp_double_detdict[temp_double_det] = amplitude(det,temp_double_det)

detdict = {}
detdict.update(original_detdict)
detdict.update(temp_detdict)
detdict.update(temp_double_detdict)
for i in detdict:
    print i, detdict[i]
print sorted(detdict.items(), key=lambda x: x[1])
print len(detdict)
