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

#determine degree of excitation between two dets (as strings of {0,1})
def n_excit(idet,jdet):
    if idet==jdet:
        return 0
    return (bin(int(idet,2)^int(jdet,2)).count('1'))/2

#return alpha/beta elements of det (either as a string or as a list of one-char strings)
def alpha(det):
    return det[::2]
def beta(det):
    return det[1::2]

def occ_alpha(detstring):
    alphastring = alpha(detstring)
    occlist=[]
    



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
h1e = reduce(np.dot, (c.T, myhf.get_hcore(), c))
eri = ao2mo.kernel(mol, c)
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


fulldetlist=[] #mapping from det number to bitstring (consider using int instead of string? or two ints/strings for alpha and beta spin)
detid={} #mapping from bitstring to det number (not sure whether this will be necessary, but might make things easier for now)
detcount=0
for occlist in itertools.combinations(range(num_orbs),num_occ):
    idet=["0" for i in range(num_orbs)]
    for orb in occlist:
        idet[orb]="1"
    fulldetlist.append(''.join(idet))
    detid[''.join(idet)]=detcount
    detcount+=1
ndets=int(sp.special.binom(num_orbs,num_occ))
#start with HF determinant
original_detdict = {fulldetlist[0]:1.0}
#lists for csr sparse storage of hamiltonian
#if this is just for storage (and not diagonalization) then we can use a dict instead (or store as upper half of sparse matrix)
hrow=[]
hcol=[]
hval=[]
for i in range(ndets):
    hii=0.0 #placeholder: generate hii here

    idet=fulldetlist[i]
    ideta=alpha(idet)
    idetb=beta(idet)
#might be better to just loop over occupied orbs instead of over all indices
    for ii, (ia, ib) in enumerate(zip(ideta,idetb)):
        if "1" in ia+ib:
            hii+= (int(ia)+int(ib))*h1e[ii,ii]
            jii=eri[idx2(ii,ii),idx2(ii,ii)]
            if ia+ib == "11":
                hii += jii
            for jj,ja,jb in zip(range(ii+1,nao),ideta[ii+1:],idetb[ii+1:]):
                jij=eri[idx2(ii,ii),idx2(jj,jj)]
                kij=eri[idx2(ii,jj),idx2(jj,ii)]
                if ia+jb=="11":
                    hii += jij
                if ib+ja=="11":
                    hii += jij
                if ia+ja=="11":
                    hii += jij
                    hii -= kij
                if ib+jb=="11":
                    hii += jij
                    hii -= kij
    hrow.append(i)
    hcol.append(i)
    hval.append(hii)
#    for j in range(i+1,ndets):
#
#        jdet=fulldetlist[j]
#        nexc_ij = n_excit(idet,jdet)
#        if nexc_ij <= 2:
#            jdeta=alpha(jdet)
#            jdetb=beta(jdet)
#            hij=0.0 #placeholder: generate hij here
#            if nexc_ij==2:
#                (hole1,hole2,part1,part2,sign12)=get_double_exc(idet,jdet)
#                hij += sign12 * (eri[idx2(part1,hole1),idx2(part2,hole2)] - eri[idx2(part1,hole2),idx2(part2,hole1)])
#            else:
#                (hole1,part1,sign1,occsame,occdiff) = get_single_exc(idet,jdet)
#                samespin = (hole1%2 == part1%2)
#                if samespin:
#                    hij += h1e[hole1/2,part1/2]
#                for ii in occsame:
#                    hij += eri[idx2(part1
#
#
#
#                
#            hrow.append(i)
#            hrow.append(j)
#            hcol.append(j)
#            hcol.append(i)
#            hval.append(hij)
#            hval.append(hij)
fullham=sp.sparse.csr_matrix((hval,(hrow,hcol)),shape=(ndets,ndets))
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
