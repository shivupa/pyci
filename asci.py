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
num_orbs=2*len(h1e)
num_occ = mol.nelectron
num_virt = ((np.shape(h1e)[0]*2)-mol.nelectron)
bitstring = "1"*num_occ
bitstring += "0"*num_virt
print(bitstring)
starting_amplitude =1.0
original_detdict = {bitstring:starting_amplitude}

H_core = np.array((cdets,cdets))
H_target = np.array((tdets,tdets))
#generate all determinants
fulldetlist=[]
print(num_orbs)
for occlist in itertools.combinations(range(num_orbs),num_occ):
    idet=["0" for i in range(num_orbs)]
    for orb in occlist:
        idet[orb]="1"
    fulldetlist.append(''.join(idet))
ndets=sp.special.binom(num_orbs,num_occ)
#a,b,c lists for csr sparse hamiltonian
#row,col,value?
a=[]
b=[]
c=[]
for i in range(ndets):
    hii=1.0 #placeholder: generate hii here
    a.append(i)
    b.append(i)
    c.append(hii)
    for j in range(i+1,ndets):
        hij=1.0 #placeholder: generate hij here
        a.append(i)
        a.append(j)
        b.append(j)
        b.append(i)
        c.append(hij)
        c.append(hij)
fullham=sp.sparse.csr_matrix((c,(a,b)),shape=(ndets,ndets))
print(len(fulldetlist))


#############
# MAIN LOOP
#############
# a^dagger_i a_j |psi>
temp_detdict = {}
temp_double_detdict = {}
new_detdict = copy.deepcopy(original_detdict)
print(temp_detdict)

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
    print(occ_index)
    print(virt_index)
    for i in occ_index:
        for j in virt_index:
            temp_det = list(det)
            temp_det[i] = "0"
            temp_det[j] = "1"
            temp_det =  ''.join(temp_det)
            temp_detdict[temp_det] = 0.1
            #print temp_det, temp_amplitude
            for k in occ_index:
                for l in virt_index:
                    if k>i and l>j:
                        temp_double_det = list(det)
                        temp_double_det[i] = "0"
                        temp_double_det[j] = "1"
                        temp_double_det[k] = "0"
                        temp_double_det[l] = "1"
                        temp_double_det =  ''.join(temp_double_det)
                        temp_double_detdict[temp_double_det] = 0.3
for i in temp_detdict:
    try:
        new_detdict[i] += temp_detdict[i]
    except:
        new_detdict.update({i:temp_detdict[i]})
for i in temp_double_detdict:
    try:
        new_detdict[i] += temp_double_detdict[i]
    except:
        new_detdict.update({i:temp_double_detdict[i]})
#new_detdict.update(temp_double_detdict)
#detdict = {}
#new_detdict.update(original_detdict)
#print("shiv",len(temp_detdict))
#print("shiv",len(temp_double_detdict))
for i in new_detdict:
    print(i, new_detdict[i])
print(sorted(new_detdict.items(), key=lambda x: x[1]))
print(len(new_detdict))
