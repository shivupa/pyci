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
#TODO: only cache i<j elements?
__idx2_cache = {}
def idx2(i,j):
    if (i,j) in __idx2_cache:
        return __idx2_cache[i,j]
    elif i>j:
        __idx2_cache[i,j] = int(i*(i+1)/2+j)
    else:
        __idx2_cache[i,j] = int(j*(j+1)/2+i)
    return __idx2_cache[i,j]

#not sure whether caching is worthwhile here if this just calls idx2, which is already cached 
#__idx4_cache = {}
def idx4(i,j,k,l):
    """idx4(i,j,k,l) returns 2-tuple corresponding to (ij|kl) in 
    square eri array (size n*(n-1)/2 square) (4-fold symmetry?)"""
#    if (i,j,k,l) in __idx4_cache:
    return (idx2(i,j),idx2(k,l))

#determine degree of excitation between two dets (as strings of {0,1})
def n_excit(idet,jdet):
    """get the hamming weight of a bitwise xor on two determinants.
    This will show how the number of orbitals in which they have different 
    occupations"""
    if idet==jdet:
        return 0
    aexc = n_excit_spin(idet,jdet,0)
    bexc = n_excit_spin(idet,jdet,1)
    return aexc+bexc

def n_excit_spin(idet,jdet,spin):
    """get the hamming weight of a bitwise xor on two determinants.
    This will show how the number of orbitals in which they have different 
    occupations"""
    if idet[spin]==jdet[spin]:
        return 0
    return (bin(int(idet[spin],2)^int(jdet[spin],2)).count('1'))/2 

#get hamming weight
#technically, this is the number of nonzero bits in a binary int, but we might be using strings
def hamweight(strdet):
    return strdet.count('1')

def bitstr2intlist(detstr):
    """turn a string into a list of ints
    input of "1100110" will return [1,1,0,0,1,1,0]"""
    return list(map(int,list(detstr)))

def gen_dets(norb,na,nb):
    """generate all determinants with a given number of spatial orbitals 
    and alpha,beta electrons.
    return a list of 2-tuples of strings"""
    adets=[]
    #loop over all subsets of size na from the list of orbitals
    for alist in itertools.combinations(range(norb),na):
        #start will all orbs unoccupied
        idet=["0" for i in range(norb)]
        for orb in alist:
            #for each occupied orbital (index), replace the "0" with a "1"
            idet[orb]="1"
        #turn the list into a string
        adets.append(''.join(idet))
    if na==nb:
        #if nb==na, make a copy of the alpha strings (beta will be the same)
        bdets=adets[:]
    else:
        bdets=[]
        for blist in itertools.combinations(range(norb),nb):
            idet=["0" for i in range(norb)]
            for orb in blist:
                idet[orb]="1"
            bdets.append(''.join(idet))
    #return all pairs of (alpha,beta) strings
    return [(i,j) for i in adets for j in bdets]

def d_a_b_occ(idet):
    """given idet as a 2-tuple of alpha,beta bitstrings, 
    return 3-tuple of lists of indices of 
    doubly-occupied, singly-occupied (alpha), singly-occupied (beta) orbitals"""
    docc = []
    aocc = []
    bocc = []
#make two lists of ints so we can use binary logical operators on them
    aint,bint = map(bitstr2intlist,idet)
    for i, (a, b) in enumerate(zip(aint,bint)):
        if a & b:
            #if alpha and beta, then this orbital is doubly occupied
            docc.append(i)
        elif a & ~b:
            #if alpha and not beta, then this is singly occupied (alpha)
            aocc.append(i)
        elif b & ~a:
            #if beta and not alpha, then this is singly occupied (beta)
            bocc.append(i)
    return (docc,aocc,bocc)

def hole_part_sign_single(idet,jdet,spin):
    holeint,partint = map(bitstr2intlist,(idet[spin],jdet[spin]))
    sign=0 #keep track of parity
    perm=False
    #to be used in keeping track of parity
    if idet[spin]<jdet[spin]:
        order=1 #we're going to get to the particle first
    else:
        order=0 #we're going to get to the hole first

    for i, (h, p) in enumerate(zip(holeint,partint)):
        #if only i is occupied, this is the particle orbital
        if h & ~p:
            hole=i
            if order==0:
                #start keeping track of permutation parity
                perm=True
                sign=1
            else:
                #stop keeping track of parity
                perm=False
        elif p & ~h:
            part=i
            if order==0:
                perm=False
            else:
                perm=True
                sign=1
        elif perm==True:
            #if we're keeping track of parity (i.e. if we're between hole and particle indices)
            if p: #if p & h
                #if orb is occupied in both dets, change sign of parity
                sign *= -1
    return (hole,part,sign)

def d_a_b_1hole(idet,hole,spin):
    #get doubly/singly occ orbs in the first det            
    docc,aocc,bocc = d_a_b_occ(idet)

    #correct for the excitation to get only the orbs that are occupied in both dets
    if hole in docc:
        docc = sorted(list(set(docc)-{hole}))
        if spin==1:
            aocc.append(hole)
        else:
            bocc.append(hole)
    elif spin==0:
        aocc = sorted(list(set(aocc)-{hole}))
    else:
        bocc = sorted(list(set(bocc)-{hole}))
    return (docc,aocc,bocc)

def d_a_b_single(idet,jdet):
    #if alpha strings are the same for both dets, the difference is in the beta part
    #alpha is element 0, beta is element 1
    if idet[0]==jdet[0]:
        spin=1
    else:
        spin=0
    hole,part,sign = hole_part_sign_single(idet,jdet,spin)
    docc,aocc,bocc = d_a_b_1hole(idet,hole,spin)
    return (hole,part,sign,spin,docc,aocc,bocc)
        
def d_a_b_double(idet,jdet):
    pass
# Hii in spinorbs: 
# sum_i^{occ} <i|hcore|i> + 1/2 sum_{i,j}^{occ} (ii|jj) - (ij|ji)

# Hii in spatial orbs:
#  1-electron terms:
# sum_i^{singly-occ} <i|hcore|i>
# + sum_i^{doubly-occ} 2 * <i|hcore|i>

#   2-electron terms:
# double double:              2 * (ii|jj) - (ij|ji)
# single single parallel:     0.5 * ((ii|jj) - (ij|ji))
# single single antiparallel: 0.5 * (ii|jj)
# double single:              (ii|jj) - 0.5 * (ij|ji)

#TODO: test to make sure there aren't any missing factors of 2 or 0.5
#TODO: just use alpha/beta occ lists instead of double/single occ lists
def calc_hii(idet,hcore,eri):
    hii=0.0
    docc,aocc,bocc = d_a_b_occ(idet)
    for si in aocc+bocc:
        hii += hcore[si,si]
    for di in docc:
        hii += 2.0 * hcore[di,di]
        for dj in docc:
            hii += 2.0 * eri[idx4(di,di,dj,dj)]
            hii -= eri[idx4(di,dj,dj,di)]
        for si in aocc+bocc:
            hii += 2.0 * eri[idx4(dj,dj,si,si)]
            hii -= eri[idx4(dj,si,si,dj)]
    for ai in aocc:
        for aj in aocc:
            hii += 0.5 * eri[idx4(ai,ai,aj,aj)]
            hii -= 0.5 * eri[idx4(ai,aj,aj,ai)]
    for bi in bocc:
        for bj in bocc:
            hii += 0.5 * eri[idx4(bi,bi,bj,bj)]
            hii -= 0.5 * eri[idx4(bi,bj,bj,bi)]
    for ai in aocc:
        for bj in bocc:
            hii += 0.5 * eri[idx4(ai,ai,bj,bj)]
    return hii
# Hij(a->r) in spinorbs:
# <r|hcore|i> + sum_j^{occ(both)} (ri|jj) - (rj|ji)
# multiply by appropriate sign
# (parity of permutation that puts orbitals back in normal order from direct hole->particle substitution)
def calc_hij_single(idet,jdet,hcore,eri):
    hij=0.0
    hole,part,sign,spin,docc,aocc,bocc = d_a_b_single(idet,jdet)
    hij += hcore[part,hole]
    for di in docc:
        hij += 2.0 * eri[idx4(part,hole,di,di)]
        hij -= eri[idx4(part,di,di,hole)]
    for si in (aocc,bocc)[spin]:
        hij += eri[idx4(part,hole,si,si)]
        hij -= eri[idx4(part,si,si,hole)]
    for si in (bocc,aocc)[spin]:
        hij += eri[idx4(part,hole,si,si)]
    return hij


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
h1e = reduce(np.dot, (c.T, myhf.get_hcore(), c))
eri = ao2mo.kernel(mol, c)
cdets = 25
tdets = 50
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


fulldetlist=gen_dets(nao,Na,Nb) 
ndets=len(fulldetlist)
#start with HF determinant
original_detdict = {fulldetlist[0]:1.0}
#lists for csr sparse storage of hamiltonian
#if this is just for storage (and not diagonalization) then we can use a dict instead (or store as upper half of sparse matrix)
hrow=[]
hcol=[]
hval=[]
for i in range(ndets):
    idet=fulldetlist[i]
    hii = calc_hii(idet,h1e,eri)
    hrow.append(i)
    hcol.append(i)
    hval.append(hii)
    for j in range(i+1,ndets):
        jdet=fulldetlist[j]
        nexc_ij = n_excit(idet,jdet)
        if nexc_ij in (1,2):
            if nexc_ij==1:
                hij = calc_hij_single(idet,jdet,h1e,eri)
            else:
                hij=2.0
            #    hij = calc_hij_double
            hrow.append(i)
            hrow.append(j)
            hcol.append(j)
            hcol.append(i)
            hval.append(hij)
            hval.append(hij)
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
