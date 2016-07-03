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


mol = gto.M(
    atom = [['O', (0.000000000000,  -0.143225816552,   0.000000000000)],
            ['H', (1.638036840407,   1.136548822547,  -0.000000000000)],
            ['H', (-1.638036840407,   1.136548822547,  -0.000000000000)]],
    basis = 'STO-3G',
    verbose = 1,
    unit='b'
)
Na,Nb = mol.nelec #nelec is a tuple with (N_alpha, N_beta)
nao=mol.nao_nr()
s = mol.intor('cint1e_ovlp_sph')
t = mol.intor('cint1e_kin_sph')
v = mol.intor('cint1e_nuc_sph')
h=t+v
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
#if you change the sign of these two orbitals, the hamiltonian matrix elements agree with those from GAMESS
#c.T[2]*=-1
#c.T[5]*=-1
cisolver = fci.FCI(mol, c)
print('PYSCF  E(FCI) = %.12f' % (cisolver.kernel()[0] + mol.energy_nuc()))
efci = cisolver.kernel(nroots=15)[0] + mol.energy_nuc()
h1e = reduce(np.dot, (c.T, myhf.get_hcore(), c))
eri = ao2mo.kernel(mol, c)
cdets = 25
tdets = 50
threshold = 1e-13 #threshold for hii and hij
#use eri[idx2(i,j),idx2(k,l)] to get (ij|kl) chemists' notation 2e- ints

#make full 4-index eris in MO basis (only for testing idx2)
#eri_mo = ao2mo.restore(1, eri, nao)

#eri in AO basis
#eri_ao = mol.intor('cint2e_sph')
#eri_ao = eri_ao.reshape([nao,nao,nao,nao])

#print h1e
#print eri
#print np.shape(h1e),np.shape(eri)
#print mol.nelectron, np.shape(h1e)[0]*2
num_orbs=2*nao
num_occ = mol.nelectron
num_virt = num_orbs - num_occ
#bitstring = "1"*num_occ
#bitstring += "0"*num_virt
#print(bitstring)
#starting_amplitude =1.0
#original_detdict = {bitstring:starting_amplitude}

H_core = np.array((cdets,cdets))
H_target = np.array((tdets,tdets))

#generate all determinants


fulldetlist_sets=gen_dets_sets(nao,Na,Nb)
ndets=len(fulldetlist_sets)
#start with HF determinant
original_detdict = {fulldetlist_sets[0]:1.0}
#lists for csr sparse storage of hamiltonian
#if this is just for storage (and not diagonalization) then we can use a dict instead (or store as upper half of sparse matrix)
hrow=[]
hcol=[]
hval=[]
for i in range(ndets):
    idet=fulldetlist_sets[i]
    hii = calc_hii_sets(idet,h1e,eri)
    if abs(hii)>threshold: #we probably don't need this
        hrow.append(i)
        hcol.append(i)
        hval.append(hii)
    for j in range(i+1,ndets):
        jdet=fulldetlist_sets[j]
        nexc_ij = n_excit_sets(idet,jdet)
        if nexc_ij in (1,2):
            if nexc_ij==1:
                hij = calc_hij_single_sets(idet,jdet,h1e,eri)
            else:
                hij = calc_hij_double_sets(idet,jdet,h1e,eri)
            if abs(hij)>threshold:
                hrow.append(i)
                hrow.append(j)
                hcol.append(j)
                hcol.append(i)
                hval.append(hij)
                hval.append(hij)
fullham=sp.sparse.csr_matrix((hval,(hrow,hcol)),shape=(ndets,ndets))
#hamiltonian_heatmap(fullham);
print(len(fulldetlist_sets))
eig_vals,eig_vecs = sp.sparse.linalg.eigsh(fullham,k=30)
eig_vals_sorted = sorted(eig_vals)[:15] + mol.energy_nuc()
eig_vals_gamess = [-75.0129802245,
                   -74.7364625517,
                   -74.6886742417,
                   -74.6531877287]
print("pyci eigvals vs PYSCF eigvals")
for i,j in zip(eig_vals_sorted, efci):
    print(i,j)

#print("pyci matrix elements vs GAMESS matrix elements")
#print("hii(2222200) = ",fullham[0,0] + mol.energy_nuc())
#print("GAMESS energy = -74.9420799538 ")
#print("hii(2222020) = ",fullham[22,22] + mol.energy_nuc())
#print("GAMESS energy = -73.9922866074 ")

badlist=[]
goodlist=[]
#ham-0: 15 ok (these are dets with no singly occupied orbs)
#ham-0: 284 not ok (these all have some singly occupied orbs)
#ham-0: (double excitations out of lowest orb are not output by gamess?)

#ham-1: 266 ok nonzero
#ham-1: 1736 ok zero
#ham-1: 464 with wrong sign

#ham-2: 1942 ok nonzero
#ham-2: 9544 ok zero
#ham-2: 1744 with wrong sign
with open("./h2o-ref/ham-2","r") as f:
    for line in f:
        numbers_str = line.split()
#        print(numbers_str)
        a1occ=frozenset({int(i)-1 for i in numbers_str[0:5]})
        b1occ=frozenset({int(i)-1 for i in numbers_str[5:10]})
        a2occ=frozenset({int(i)-1 for i in numbers_str[10:15]})
        b2occ=frozenset({int(i)-1 for i in numbers_str[15:20]})
#        print(a1occ,b1occ,a2occ,b2occ)
        val=float(numbers_str[20])
        det1=(a1occ,b1occ)
        det2=(a2occ,b2occ)
        hij=55.0
        nexc=n_excit_sets(det1,det2)
        nexca=n_excit_spin_sets(det1,det2,0)
        nexcb=n_excit_spin_sets(det1,det2,1)
        if nexc==0:
            hij=calc_hii_sets(det1,h1e,eri)
        elif nexc==1:
            hij=calc_hij_single_sets(det1,det2,h1e,eri)
        elif nexc==2:
            hij=calc_hij_double_sets(det1,det2,h1e,eri)
        if abs(val-hij) > 0.0000001:
            badlist.append((det1,det2,nexc,nexca,nexcb,val,hij))
        else:
            goodlist.append((det1,det2,nexc,nexca,nexcb,val,hij))
good1=[i for i in goodlist if i[2]==1]
good2=[i for i in goodlist if i[2]==2]
bad1=[i for i in badlist if i[2]==1]
bad2=[i for i in badlist if i[2]==2]

goodzero=[i for i in goodlist if i[5]==0.0]
goodfinite=[i for i in goodlist if i[5]!=0.0]
badsign=[i for i in badlist if abs(i[5]+i[6])<0.0000001]
badabsval=[i for i in badlist if abs(i[5]+i[6])>=0.0000001]
badsignholepart=[(
    (list(i[0][0]-i[1][0]),list(i[1][0]-i[0][0]),list(i[0][0] & i[1][0])),
    (list(i[0][1]-i[1][1]),list(i[1][1]-i[0][1]),list(i[0][1] & i[1][1])),
    i[2],i[3],i[4],i[5],i[6]) for i in badsign]
print("badsign: ",len(badsign))
print("badlist: ",len(badlist))
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

#one of these agrees with gamess and one does not
#print("d_a_b_single(('1111100','1110110'),('1111100','1111100'))")
#d_a_b_single(('1111100','1110110'),('1111100','1111100'))

#print("d_a_b_single(('1111100','1011110'),('1111100','1110110'))")
#print(d_a_b_single(('1111100','1011110'),('1111100','1110110')))

#print("d_a_b_single(('1111100','1110011'),('1111100','1111001'))")
#print(d_a_b_single(('1111100','1110011'),('1111100','1111001')))
