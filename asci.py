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
#############
# INPUT
#############
#TODO: implement function that finds particles/holes based on set operations (will be easier with aocc,bocc lists of indices instead of docc,aocc(single),bocc(single)

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

def occ2bitstr(occlist,norb,index=0):
    bitlist=["0" for i in range(norb)]
    for i in occlist:
        bitlist[i-index]="1"
    return ''.join(bitlist)



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

def a_b_occ(idet):
    """given idet as a 2-tuple of alpha,beta bitstrings,
    return 2-tuple of lists of indices of
    occupied alpha and beta orbitals"""
    aocc = []
    bocc = []
    aint,bint = map(bitstr2intlist,idet)
    for i, (a, b) in enumerate(zip(aint,bint)):
        if a:
            aocc.append(i)
        if b:
            bocc.append(i)
    return (aocc,bocc)

def hole_part_sign_single(idet,jdet,spin,debug=False):
    holeint,partint = map(bitstr2intlist,(idet[spin],jdet[spin]))
    if debug:
        print(holeint,partint)
    for i, (h, p) in enumerate(zip(holeint,partint)):
        #if only i is occupied, this is the particle orbital
        if debug:
            print(i,h,p)
        if h & ~p:
            hole=i
            if debug:
                print("hole = ",i)
        elif p & ~h:
            part=i
            if debug:
                print("part = ",i)
    sign = getsign(holeint,partint,hole,part)
    return (hole,part,sign)

def signstr(s1,s2,h1,p1):
    holeint,partint = map(bitstr2intlist,(s1,s2))
    print(holeint)
    print(partint)
    return getsign(holeint,partint,h1,p1,debug=True)

def holes_parts_sign_double(idet,jdet,spin):
    holeint,partint = map(bitstr2intlist,(idet[spin],jdet[spin]))
    holes=[]
    parts=[]
    for i, (h,p) in enumerate(zip(holeint,partint)):
        if h & ~p:
            holes.append(i)
        elif p & ~h:
            parts.append(i)
    h1,h2 = holes
    p1,p2 = parts
    ###DEBUG DEBUG DEBUG
    if(h1>h2):
        print("DEBUG: h1>h2")
    if(p1>p2):
        print("DEBUG: p1>p2")
    ####################
    sign1 = getsign(holeint,partint,h1,p1)
    sign2 = getsign(holeint,partint,h2,p2)
    return (h1,h2,p1,p2,sign1*sign2)

def getsign(holeint,partint,h,p,debug=False):

    #determine which index comes first (hole or particle) for each pair
    if h < p:
        stri = holeint[h:p]
        strj = partint[h:p]
    else:
        stri = holeint[p:h]
        strj = partint[p:h]
    sign=1
    if debug:
        print(stri,strj)
    for i,j in zip(stri,strj):
        if debug:
            print(i,j)
        if i & j:
            if debug:
                print ("signchange")
            sign *= -1
    return sign




def hole_part_sign_spin_double(idet,jdet):
    #if the two excitations are of different spin, just do them individually
    x0 = n_excit_spin(idet,jdet,0)
    if x0==1:
        samespin=False
        hole1,part1,sign1 = hole_part_sign_single(idet,jdet,0)
        hole2,part2,sign2 = hole_part_sign_single(idet,jdet,1)
        sign = sign1 * sign2
    else:
        samespin=True
        if x0==0:
            spin = 1
        else:
            spin = 0
        #TODO get holes, particles, and sign
        hole1,hole2,part1,part2,sign = holes_parts_sign_double(idet,jdet,spin)
    return (hole1,hole2,part1,part2,sign,samespin)

def d_a_b_1hole(idet,hole,spin):
    #get doubly/singly occ orbs in the first det
    docc,aocc,bocc = d_a_b_occ(idet)

    #account for the excitation to obtain only the orbs that are occupied in both dets
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
#    if (abs(hii)>threshold):
#        return hii
#    else:
#        return 0.0
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
    hij *= sign
#    if (abs(hij)>threshold):
#        return hij
#    else:
#        return 0.0
    return hij

def calc_hij_double(idet,jdet,hcore,eri):
    hij=0.0
    h1,h2,p1,p2,sign,samespin = hole_part_sign_spin_double(idet,jdet)
    hij += eri[idx4(p1,h1,p2,h2)]
    if samespin:
        hij -= eri[idx4(p1,h2,p2,h1)]
    hij *= sign
#    if (abs(hij)>threshold):
#        return hij
#    else:
#        return 0.0
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
cisolver = fci.FCI(mol, c)
print('PYSCF  E(FCI) = %.12f' % (cisolver.kernel()[0] + mol.energy_nuc()))
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
    if abs(hii)>threshold: #we probably don't need this
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
                hij = calc_hij_double(idet,jdet,h1e,eri)
            if abs(hij)>threshold:
                hrow.append(i)
                hrow.append(j)
                hcol.append(j)
                hcol.append(i)
                hval.append(hij)
                hval.append(hij)
fullham=sp.sparse.csr_matrix((hval,(hrow,hcol)),shape=(ndets,ndets))
print(len(fulldetlist))
eig_vals,eig_vecs = sp.sparse.linalg.eigsh(fullham,k=10)
eig_vals_sorted = sorted(eig_vals)[:4] + mol.energy_nuc()
eig_vals_gamess = [-75.0129802245,
                   -74.7364625517,
                   -74.6886742417,
                   -74.6531877287]
print("pyci eigvals vs GAMESS eigvals")
for i,j in zip(eig_vals_sorted, eig_vals_gamess):
    print(i,j)

print("pyci matrix elements vs GAMESS matrix elements")
print("hii(2222200) = ",fullham[0,0] + mol.energy_nuc())
print("GAMESS energy = -74.9420799538 ")
print("hii(2222020) = ",fullham[22,22] + mol.energy_nuc())
print("GAMESS energy = -73.9922866074 ")

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
with open("./h2o-ref/ham-1","r") as f:
    for line in f:
        numbers_str = line.split()
#        print(numbers_str)
        a1occ=occ2bitstr(map(int,numbers_str[0:5]),7,1)
        b1occ=occ2bitstr(map(int,numbers_str[5:10]),7,1)
        a2occ=occ2bitstr(map(int,numbers_str[10:15]),7,1)
        b2occ=occ2bitstr(map(int,numbers_str[15:20]),7,1)
#        print(a1occ,b1occ,a2occ,b2occ)
        val=float(numbers_str[20])
        det1=(a1occ,b1occ)
        det2=(a2occ,b2occ)
        hij=55.0
        nexc=n_excit(det1,det2)
        nexca=n_excit_spin(det1,det2,0)
        nexcb=n_excit_spin(det1,det2,1)
        if nexc==0:
            hij=calc_hii(det1,h1e,eri)
        elif nexc==1:
            hij=calc_hij_single(det1,det2,h1e,eri)
        elif nexc==2:
            hij=calc_hij_double(det1,det2,h1e,eri)
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
