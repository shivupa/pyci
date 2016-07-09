import itertools
import scipy as sp
import scipy.linalg as spla
import scipy.sparse.linalg as splinalg
import copy
import numpy as np
from functools import reduce
import pyscf
import itertools
import h5py
from pyscf import gto, scf, ao2mo, fci
import pyscf.tools as pt
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
#can be reformulated with sets. will be easier

def n_excit(idet,jdet):
    """get the hamming weight of a bitwise xor on two determinants.
    This will show how the number of orbitals in which they have different
    occupations"""
    if idet==jdet:
        return 0
    aexc = n_excit_spin(idet,jdet,0)
    bexc = n_excit_spin(idet,jdet,1)
    return aexc+bexc

def n_excit_sets(idet,jdet):
    if idet==jdet:
        return 0
    aexc = n_excit_spin_sets(idet,jdet,0)
    bexc = n_excit_spin_sets(idet,jdet,1)
    return aexc+bexc

def n_excit_spin(idet,jdet,spin):
    """get the hamming weight of a bitwise xor on two determinants.
    This will show how the number of orbitals in which they have different
    occupations"""
    if idet[spin]==jdet[spin]:
        return 0
    return (bin(int(idet[spin],2)^int(jdet[spin],2)).count('1'))/2

def n_excit_spin_sets(idet,jdet,spin):
    if idet[spin]==jdet[spin]:
        return 0
    return len(idet[spin]-jdet[spin])

#get hamming weight
#technically, this is the number of nonzero bits in a binary int, but we might be using strings\
#long slow down if we call this in n_excit_spin weird
def hamweight(strdet):
    return strdet.count('1')
def bitstr2intlist(detstr):
    """turn a string into a list of ints
    input of "1100110" will return [1,1,0,0,1,1,0]"""
    return list(map(int,list(detstr)))
def occ2bitstr(occlist,norb,index=0):
    """turn a list of ints of indices of occupied orbitals
    and total number of orbitals into a bit string """
    bitlist=["0" for i in range(norb)]
    for i in occlist:
        bitlist[i-index]="1"
    return ''.join(bitlist)
def gen_dets_sets(norb,na,nb):
    """generate all determinants with a given number of spatial orbitals
    and alpha,beta electrons.
    return a list of 2-tuples of strings"""
    adets=[]
    #loop over all subsets of size na from the list of orbitals
    for alist in itertools.combinations(range(norb),na):
        #start will all orbs unoccupied
        adets.append(frozenset(alist))
    if na==nb:
        #if nb==na, make a copy of the alpha strings (beta will be the same)
        bdets=adets[:]
    else:
        bdets=[]
        for blist in itertools.combinations(range(norb),nb):
            bdets.append(frozenset(blist))
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

def hole_part_sign_single_sets(idet,jdet,spin,debug=False):
    holeset,partset = (idet[spin],jdet[spin])
    if debug:
        print(holeset,partset)
    hole,=(holeset-partset)
    part,=(partset-holeset)
    sign = getsign_sets(holeset,hole,part)
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
    sign1 = getsign(holeint,partint,h1,p1)
    sign2 = getsign(holeint,partint,h2,p2)
    return (h1,h2,p1,p2,sign1*sign2)

def holes_parts_sign_double_sets(idet,jdet,spin):
    holeset,partset = (idet[spin],jdet[spin])
    holes=set(holeset-partset)
    parts=set(partset-holeset)
    h1 = holes.pop()
    h2 = holes.pop()
    p1 = parts.pop()
    p2 = parts.pop()
    sign1 = getsign_sets(holeset,h1,p1)
    sign2 = getsign_sets(partset,h2,p2)
    sign = sign1*sign2
    return (h1,h2,p1,p2,sign)

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

def getsign_sets(holeset,h,p,debug=False):
    #determine which index comes first (hole or particle) for each pair
    if h < p:
        num = [i for i in holeset if i<p and i>h]
    else:
        num = [i for i in holeset if i<h and i>p]
    sign=(-1)**len(num)
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

def hole_part_sign_spin_double_sets(idet,jdet):
    #if the two excitations are of different spin, just do them individually
    x0 = n_excit_spin_sets(idet,jdet,0)
    if x0==1:
        samespin=False
        hole1,part1,sign1 = hole_part_sign_single_sets(idet,jdet,0)
        hole2,part2,sign2 = hole_part_sign_single_sets(idet,jdet,1)
        sign = sign1 * sign2
    else:
        samespin=True
        if x0==0:
            spin = 1
        else:
            spin = 0
        #TODO get holes, particles, and sign
        hole1,hole2,part1,part2,sign = holes_parts_sign_double_sets(idet,jdet,spin)
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

def a_b_1hole_sets(idet,hole,spin):
    if spin==0:
        return (idet[0]-{hole},idet[1])
    else:
        return (idet[0],idet[1]-{hole})

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

def a_b_single_sets(idet,jdet):
    #if alpha strings are the same for both dets, the difference is in the beta part
    #alpha is element 0, beta is element 1
    if idet[0]==jdet[0]:
        spin=1
    else:
        spin=0
    hole,part,sign = hole_part_sign_single_sets(idet,jdet,spin)
    aocc,bocc = a_b_1hole_sets(idet,hole,spin)
    return (hole,part,sign,spin,aocc,bocc)
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
""" #TODO(shiv): Unused consider deleting? We may eventually need this though
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
    return hij
def calc_hij_double(idet,jdet,hcore,eri):
    hij=0.0
    h1,h2,p1,p2,sign,samespin = hole_part_sign_spin_double(idet,jdet)
    hij += eri[idx4(p1,h1,p2,h2)]
    if samespin:
        hij -= eri[idx4(p1,h2,p2,h1)]
    hij *= sign
    return hij
"""
# Hij(a->r) in spinorbs:
# <r|hcore|i> + sum_j^{occ(both)} (ri|jj) - (rj|ji)
# multiply by appropriate sign
# (parity of permutation that puts orbitals back in normal order from direct hole->particle substitution)

def calc_hii_sets(idet,hcore,eri):
    hii=0.0
    aocc,bocc = idet
    for ia in aocc:
        hii += hcore[ia,ia]
    for ib in bocc:
        hii += hcore[ib,ib]
    for ia in aocc:
        for ja in aocc:
            hii += 0.5*(eri[idx4(ia,ia,ja,ja)]-eri[idx4(ia,ja,ja,ia)])
    for ib in bocc:
        for jb in bocc:
            hii += 0.5*(eri[idx4(ib,ib,jb,jb)]-eri[idx4(ib,jb,jb,ib)])
    for ia in aocc:
        for jb in bocc:
            hii += eri[idx4(ia,ia,jb,jb)]
    return hii
def calc_hij_single_sets(idet,jdet,hcore,eri):
    hij=0.0
    hole,part,sign,spin,aocc,bocc = a_b_single_sets(idet,jdet)
    hij += hcore[part,hole]
    for si in (aocc,bocc)[spin]:
        hij += eri[idx4(part,hole,si,si)]
        hij -= eri[idx4(part,si,si,hole)]
    for si in (bocc,aocc)[spin]:
        hij += eri[idx4(part,hole,si,si)]
    hij *= sign
    return hij
def calc_hij_double_sets(idet,jdet,hcore,eri):
    hij=0.0
    h1,h2,p1,p2,sign,samespin = hole_part_sign_spin_double_sets(idet,jdet)
    hij += eri[idx4(p1,h1,p2,h2)]
    if samespin:
        hij -= eri[idx4(p1,h2,p2,h1)]
    hij *= sign
    return hij

def get_excitations(det,norb,aexc,bexc):
    """
    usage: get_excitations(det,norb,aexc,bexc)
    returns a list of determinants connected to det with alpha(beta) excitation level aexc(bexc)
    norb is the number of spatial orbitals
    det should be a pair (2-tuple or list) or sets of occupied orbital indices)
    returned determinants will be 2-tuples of frozensets of occupied indices (alpha and beta)
    """
    aocc=det[0]
    bocc=det[1]
    orbs=frozenset(range(norb))
    avirt=orbs-aocc
    bvirt=orbs-bocc

    adets=[]
    bdets=[]
    if aexc > len(avirt) or aexc > len(aocc) or bexc > len(bvirt) or bexc > len(bocc):
        raise
    for iocc in itertools.combinations(aocc,aexc):
        for ivirt in itertools.combinations(avirt,aexc):
            adets.append(aocc - set(iocc) | set(ivirt))
    for iocc in itertools.combinations(bocc,bexc):
        for ivirt in itertools.combinations(bvirt,bexc):
            bdets.append(bocc - set(iocc) | set(ivirt))
    return [(i,j) for i in adets for j in bdets]

def gen_singles(det,norb):
    return get_excitations(det,norb,1,0) + get_excitations(det,norb,0,1)

def gen_doubles(det,norb):
    return get_excitations(det,norb,2,0) + get_excitations(det,norb,0,2) + get_excitations(det,norb,1,1)

def gen_singles_doubles(det,norb):
    return gen_singles(det,norb) + gen_doubles(det,norb)


##############################################ASCI funcs
def gen_dets_sets_truncated(norb,cdetlist_sets):
    """generate cdets determinants with a given number of spatial orbitals
    and alpha,beta electrons.
    return a list of 2-tuples of strings"""
    return_list = []
    for i in cdetlist_sets:
        return_list.extend(gen_singles_doubles(i,norb))
    return return_list

def construct_ham_dict(coredetlist_sets,h1e,eri):
    ham_dict = {}
    ndets=len(coredetlist_sets)
    for i in range(ndets):
        idet=coredetlist_sets[i]
        hii = calc_hii_sets(idet,h1e,eri)
        ham_dict[frozenset((idet))] = hii
        for j in range(i+1,ndets):
            jdet=coredetlist_sets[j]
            nexc_ij = n_excit_sets(idet,jdet)
            if nexc_ij in (1,2):
                if nexc_ij==1:
                    hij = calc_hij_single_sets(idet,jdet,h1e,eri)
                else:
                    hij = calc_hij_double_sets(idet,jdet,h1e,eri)
                ham_dict[frozenset((idet,jdet))] = hij
    return ham_dict

def construct_hamiltonian(ndets,coredetlist_sets,h1e,eri):
    hrow = []
    hcol = []
    hval = []
    for i in range(ndets):
        idet=coredetlist_sets[i]
        hii = calc_hii_sets(idet,h1e,eri)
        hrow.append(i)
        hcol.append(i)
        hval.append(hii)
        for j in range(i+1,ndets):
            jdet=coredetlist_sets[j]
            nexc_ij = n_excit_sets(idet,jdet)
            if nexc_ij in (1,2):
                if nexc_ij==1:
                    hij = calc_hij_single_sets(idet,jdet,h1e,eri)
                else:
                    hij = calc_hij_double_sets(idet,jdet,h1e,eri)
                hrow.append(i)
                hrow.append(j)
                hcol.append(j)
                hcol.append(i)
                hval.append(hij)
                hval.append(hij)
    return sp.sparse.csr_matrix((hval,(hrow,hcol)),shape=(ndets,ndets))

def getsmallham(dets,hamdict):
    hrow=[]
    hcol=[]
    hval=[]
    ndets=len(dets)
    for i in range(ndets):
        idet = dets[i]
        hrow.append(i)
        hcol.append(i)
        hval.append(hamdict[frozenset((idet))])
        for j in range(i+1,ndets):
            jdet = dets[j]
            if frozenset((idet,jdet)) in hamdict:
                hrow.append(i)
                hrow.append(j)
                hcol.append(j)
                hcol.append(i)
                hij = hamdict[frozenset((idet,jdet))]
                hval.append(hij)
                hval.append(hij)
    return sp.sparse.csr_matrix((hval,(hrow,hcol)),shape=(ndets,ndets))

"""TO REMOVE
def get_smaller_hamiltonian(h,indices):
    hrow = []
    hcol = []
    hval = []
    for i in range(len(indices)):
        hrow.append(i)
        hcol.append(i)
        hval.append(h[indices[i],indices[i]])
        for j in range(i+1,len(indices)):
            hrow.append(i)
            hrow.append(j)
            hcol.append(j)
            hcol.append(i)
            hval.append(h[indices[i],indices[j]])
            hval.append(h[indices[i],indices[j]])
    return sp.sparse.csr_matrix((hval,(hrow,hcol)),shape=(len(indices),len(indices)))
    """

def populatehamdict(targetdetset,hamdict,h1e,eri):
    ndets = len(targetdetset)
    update_dict = dict()

    for i in targetdetset:
        if i not in hamdict:
            update_dict[frozenset((i))] = calc_hii_sets(i,h1e,eri)
            for j in targetdetset:
                if i != j:
                    nexc_ij = n_excit_sets(i,j)
                    if nexc_ij in (1,2):
                        if nexc_ij==1:
                            update_dict[frozenset([i,j])] = calc_hij_single_sets(i,j,h1e,eri)
                        else:
                            update_dict[frozenset([i,j])] = calc_hij_double_sets(i,j,h1e,eri)
    return update_dict


def asci(mol,cdets,tdets,convergence=1e-6,printroots=4,iter_min=0):
    print("PYCI")
    print("method: ASCI")
    print("convergence: ", convergence)
    print("Core Space size: ",cdets)
    print("Target Space size: ",tdets)
    print("Number of eigenvalues: ",printroots)
    print("")
    Na,Nb = mol.nelec #nelec is a tuple with (N_alpha, N_beta)
    E_nuc = mol.energy_nuc()
    nao = mol.nao_nr()
    myhf = scf.RHF(mol)
    E_hf = myhf.kernel()
    mo_coefficients = myhf.mo_coeff
    #print("starting pyscf FCI")
    #cisolver = fci.FCI(mol, mo_coefficients)
    #efci = cisolver.kernel(nroots=printroots)[0] + mol.energy_nuc()
    #print("FCI done")
    h1e = reduce(np.dot, (mo_coefficients.T, myhf.get_hcore(), mo_coefficients))
    print("transforming eris")
    eri = ao2mo.kernel(mol, mo_coefficients)
    #use eri[idx2(i,j),idx2(k,l)] to get (ij|kl) chemists' notation 2e- ints
    #make full 4-index eris in MO basis (only for testing idx2)
    #print("generating all determinants")
    #fulldetlist_sets=gen_dets_sets(nao,Na,Nb)
    #ndets=len(fulldetlist_sets)
    #full_hamiltonian = construct_hamiltonian(ndets,fulldetlist_sets,h1e,eri)
    print("constructing full Hamiltonian")
    #hamdict = construct_ham_dict(fulldetlist_sets,h1e,eri)
    hamdict = dict()

    E_old = 0.0
    E_new = E_hf
    hfdet = (frozenset(range(Na)),frozenset(range(Nb)))
    targetdetset = set()
    coreset = {hfdet}
    C = {hfdet:1.0}
    print("\nHartree-Fock Energy: ", E_hf)
    print("\nBeginning Iterations\n")
    it_num = 0
    while(np.abs(E_new - E_old) > convergence):
        #print("is hfdet in coreset? ", hfdet in coreset)
        it_num += 1
        E_old = E_new
        #print("Core Dets: ",len(coreset))
        #step 1
        targetdetset=set()
        for idet in coreset:
            targetdetset |= set(gen_singles_doubles(idet,nao))
        A = dict.fromkeys(targetdetset, 0.0)
        hamdict.update(populatehamdict((targetdetset | coreset),hamdict,h1e,eri))
        for idet in coreset:
            for jdet in targetdetset:
                nexc_ij = n_excit_sets(idet,jdet)
                if nexc_ij in (1,2): # don't bother checking anything with a zero hamiltonian element
                    try:
                        A[jdet] += hamdict[frozenset([idet,jdet])] * C[idet]
                    except:
                        A[jdet] += hamdict[frozenset([jdet,idet])] * C[idet]
        for idet in targetdetset:
            A[idet] /= (hamdict[frozenset((idet))] - E_old)
        for idet in coreset:
            if idet in A:
                if abs(A[idet]) < abs(C[idet]): # replace with the biggest again
                    A[idet] = C[idet]
            else:
                A[idet] = C[idet]
        A_sorted = sorted(list(A.items()),key=lambda i: -abs(i[1]))
        A_truncated = A_sorted[:tdets]
        #print("Target Dets: ",len(A_truncated))
        A_dets = [i[0] for i in A_truncated]
        targetham = getsmallham(A_dets,hamdict)
        eig_vals,eig_vecs = sp.sparse.linalg.eigsh(targetham,k=2*printroots)
        eig_vals_sorted = np.sort(eig_vals)[:printroots]
        E_new = eig_vals_sorted[0]
        print("Iteration {:} Energy: ".format(it_num), E_new + E_nuc)
        #step 4
        amplitudes = eig_vecs[:,np.argsort(eig_vals)[0]]
        newdet = [i for i in zip(A_dets,amplitudes)]
        C = {}
        for i in sorted(newdet,key=lambda j: -abs(j[1]))[:cdets]:
            C[i[0]] = i[1]
        coreset = set(C.keys())
        print("")
    #   print("first {:} pyci eigvals vs PYSCF eigvals".format(printroots))
    #   for i,j in zip(eig_vals_sorted + E_nuc, efci):
    #       print(i,j)
    print("first {:} pyci eigvals".format(printroots))
    for i in (eig_vals_sorted + E_nuc):
        print(i)
    print("size of hamdict:", len(hamdict))
    print("Completed!")
##############################################HBCI functions
def heatbath(det,norb,hamdict,amplitudes,epsilon,h1e,eri):
    excitation_space = set()
    for i in det:
        excitation_space |= set(gen_singles(i,norb) + gen_doubles(i,norb))
    remove_set = set()
    for i in excitation_space:
        add = False
        for j in det:
            nexc_ij = n_excit_sets(i,j)
            if nexc_ij in (1,2):
                if nexc_ij==1:
                    h = calc_hij_single_sets(i,j,h1e,eri)
                    if abs(h*amplitudes[j]) >= epsilon:
                        add = True
                        hamdict[frozenset([i,j])] = h
                else:
                    h = calc_hij_double_sets(i,j,h1e,eri)
                    if abs(h*amplitudes[j]) >= epsilon:
                        add = True
                        hamdict[frozenset([i,j])] = h
        if add == False:
            remove_set.add(i)
    return excitation_space-remove_set, hamdict
def hbci(mol,epsilon=0.01,convergence=0.01,printroots=4):
    print("PYCI")
    print("method: HBCI")
    print("convergence: ", convergence)
    print("Epsilon: ",epsilon)
    print("Number of eigenvalues: ",printroots)
    print("")
    Na,Nb = mol.nelec #nelec is a tuple with (N_alpha, N_beta)
    E_nuc = mol.energy_nuc()
    nao = mol.nao_nr()
    myhf = scf.RHF(mol)
    E_hf = myhf.kernel()
    mo_coefficients = myhf.mo_coeff
    h1e = reduce(np.dot, (mo_coefficients.T, myhf.get_hcore(), mo_coefficients))
    eri = ao2mo.kernel(mol, mo_coefficients)
    num_orbs=2*nao
    num_occ = mol.nelectron
    num_virt = num_orbs - num_occ

    E_old = 0.0
    E_new = E_hf

    hfdet = (frozenset(range(Na)),frozenset(range(Nb)))
    oldselecteddetset= {hfdet}
    C = {hfdet:1.0}
    print("Hartree-Fock Energy: ", E_hf)
    print("")
    it_num = 1
    Converged = False
    hamdict=dict()
    while( not Converged):
        newselecteddetset=set()
        newselecteddetset,hamdict_additions = heatbath(oldselecteddetset,nao,hamdict,C,epsilon,h1e,eri)
        hamdict.update(hamdict_additions)
        newselecteddetset |= oldselecteddetset
        hamdict.update(populatehamdict((newselecteddetset),hamdict,h1e,eri))
        selectedham = getsmallham(list(newselecteddetset),hamdict)
        eig_vals,eig_vecs = sp.sparse.linalg.eigsh(selectedham,k=2*printroots)
        eig_vals_sorted = np.sort(eig_vals)[:printroots]
        E_new = eig_vals_sorted[0]
        amplitudes = eig_vecs[:,np.argsort(eig_vals)[0]]
        newdet = [i for i in zip(newselecteddetset,amplitudes)]
        C = {}
        for i in sorted(newdet,key=lambda j: -abs(j[1])):
            C[i[0]] = i[1]
        print("Iteration {:} Energy: ".format(it_num), E_new + E_nuc)
        amount_added=abs(len(newselecteddetset)-len(oldselecteddetset))
        if amount_added < np.ceil(convergence * len(oldselecteddetset)):
            Converged = True
        else:
            print("Iteration {:} space growth %: ".format(it_num), amount_added)
        oldselecteddetset |= newselecteddetset
        it_num += 1
        E_old = E_new
        print("Selected Space size: ",len(oldselecteddetset))
        print("")
    print("first {:} pyci eigvals".format(printroots))
    for i in (eig_vals_sorted + E_nuc):
        print(i)
    print("Completed!")
###########################################################
if __name__ == "__main__":
    print("\nPYCI utils file. This file was not meant to be run independently.")
