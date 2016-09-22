from __future__ import print_function

import itertools
import scipy as sp
import scipy.linalg as spla
import scipy.sparse.linalg as splinalg
import copy
import numpy as np
from functools import reduce
import itertools
from pyscf import scf, ao2mo
from pyci.methods.visualize import visualize_sets

################################################################################
# FUNCTIONS FOR CALCULATING HAMILTONIAN MATRIX ELEMENTS
################################################################################

__idx2_cache = {}
def idx2(i, j):
    """2-index transformation for accessing elements of a symmetric matrix
    stored in upper-triangular form.
    """
    if (i, j) in __idx2_cache:
        return __idx2_cache[i, j]
    elif i > j:
        __idx2_cache[i, j] = int(i*(i+1)/2+j)
    else:
        __idx2_cache[i, j] = int(j*(j+1)/2+i)
    return __idx2_cache[i, j]


def idx4(i, j, k, l):
    """idx4(i,j,k,l) returns 2-tuple corresponding to (ij|kl) in square
    ERI array (size n*(n-1)/2 square) (4-fold symmetry?)
    """
    return idx2(i, j), idx2(k, l)


def n_excit_spin_sets(idet, jdet, spin):
    """Returns the degree of excitation between two occupation lists.
    """
    if idet[spin] == jdet[spin]:
        return 0
    return len(idet[spin] - jdet[spin])


def n_excit_sets(idet, jdet):
    """Returns the degree of excitation between two determinants.
    """
    if idet == jdet:
        return 0
    aexc = n_excit_spin_sets(idet, jdet, 0)
    bexc = n_excit_spin_sets(idet, jdet, 1)
    return aexc + bexc


def gen_dets_sets(norb, na, nb):
    """Generate all determinants with a given number of spatial orbitals
    and alpha, beta electrons. Return a list of 2-tuples of strings
    corresponding to...
    """

    adets = []
    # loop over all subsets of size na from the list of orbitals
    for alist in itertools.combinations(range(norb), na):
        # start will all orbs unoccupied
        adets.append(frozenset(alist))
    if na == nb:
        # If nb == na, make a copy of the alpha strings (beta will be the same).
        bdets = adets[:]
    else:
        bdets = []
        for blist in itertools.combinations(range(norb), nb):
            bdets.append(frozenset(blist))

    # return all pairs of (alpha,beta) strings
    return [(i, j) for i in adets for j in bdets]


def hole_part_sign_single_sets(idet, jdet, spin, debug=False):
    """for two dets that differ by a single excitation, return the
    indices of hole and particle orbitals and the sign related to the
    excitation (determined by the parity of the permutation that puts
    the orbitals back in normal order after replacing the hole with
    the particle directly)
    """

    holeset, partset = idet[spin], jdet[spin]
    if debug:
        print(holeset, partset)
    hole, = holeset - partset
    part, = partset - holeset
    sign = getsign_sets(holeset, hole, part)

    return hole, part, sign


def holes_parts_sign_double_sets(idet, jdet, spin):
    """for two dets that differ by a double excitation, return the
    indices of hole and particle orbitals and the sign related to the
    excitation (determined by the parity of the permutation that puts
    the orbitals back in normal order after replacing the holes with
    the particles directly)
    """

    holeset, partset = idet[spin], jdet[spin]
    holes = set(holeset - partset)
    parts = set(partset - holeset)
    h1 = holes.pop()
    h2 = holes.pop()
    p1 = parts.pop()
    p2 = parts.pop()
    sign1 = getsign_sets(holeset, h1, p1)
    sign2 = getsign_sets(partset, h2, p2)
    sign = sign1 * sign2

    return h1, h2, p1, p2, sign


def getsign_sets(holeset, h, p):
    """For the indices of hole and particle orbitals, return the sign
    related to a single excitation.
    """
    # determine which index comes first (hole or particle) for each pair
    if h < p:
        num = [i for i in holeset if i < p and i > h]
    else:
        num = [i for i in holeset if i < h and i > p]
    sign = (-1) ** len(num)
    return sign


def hole_part_sign_spin_double_sets(idet, jdet):
    """For two determinants that differ by a double excitation, return:
    indices of hole, particle orbitals; sign related to excitation;
    boolean representing whether both excitations are of the same spin
    """
    # if the two excitations are of different spin, just do them individually
    x0 = n_excit_spin_sets(idet, jdet, 0)
    if x0 == 1:
        samespin = False
        hole1, part1, sign1 = hole_part_sign_single_sets(idet, jdet, 0)
        hole2, part2, sign2 = hole_part_sign_single_sets(idet, jdet, 1)
        sign = sign1 * sign2
    else:
        samespin = True
        if x0 == 0:
            spin = 1
        else:
            spin = 0
        hole1, hole2, part1, part2, sign = holes_parts_sign_double_sets(idet, jdet, spin)
    return hole1, hole2, part1, part2, sign, samespin


def aocc_bocc_single_sets(idet, hole, spin):
    """Deletes hole index from an occ string of a certain spin
    """
    if spin == 0:
        return idet[0] - {hole}, idet[1]
    else:
        return idet[0], idet[1] - {hole}


def hole_part_sign_spin_occ_single_sets(idet, jdet):
    """For two determinants that differ by a single excitation, return:
    indices of hole, particle orbitals; sign related to excitation
    """
    # if alpha strings are the same for both dets, the difference is in the beta part
    # alpha is element 0, beta is element 1
    if idet[0] == jdet[0]:
        spin = 1
    else:
        spin = 0
    hole, part, sign = hole_part_sign_single_sets(idet, jdet, spin)
    aocc, bocc = aocc_bocc_single_sets(idet, hole, spin)
    return hole, part, sign, spin, aocc, bocc

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

# Hij(a->r) in spinorbs:
# <r|hcore|i> + sum_j^{occ(both)} (ri|jj) - (rj|ji)
# multiply by appropriate sign
# (parity of permutation that puts orbitals back in normal order from direct hole->particle substitution)

__hamdict = {}
def calc_hii_sets(idet, hcore, eri):
    """ Calculate the diagonal hamiltonian element using the eris stored with
    4-fold symmetry
    square eri array (size n*(n-1)/2 square) (4-fold symmetry?)
    idet is a tuple of sets of occupied alpha beta orbitals
    """
    aocc, bocc = idet
    if frozenset([idet]) in __hamdict:
        return __hamdict[frozenset([idet])]
    else:
        hii = 0.0
        for ia in aocc:
            hii += hcore[ia, ia]
        for ib in bocc:
            hii += hcore[ib, ib]
        for ia in aocc:
            for ja in aocc:
                hii += 0.5 * (eri[idx4(ia, ia, ja, ja)] - eri[idx4(ia, ja, ja, ia)])
        for ib in bocc:
            for jb in bocc:
                hii += 0.5 * (eri[idx4(ib, ib, jb, jb)] - eri[idx4(ib, jb, jb, ib)])
        for ia in aocc:
            for jb in bocc:
                hii += eri[idx4(ia, ia, jb, jb)]
        __hamdict[frozenset([idet])] = hii
        return __hamdict[frozenset([idet])]


def calc_hij_single_sets(idet, jdet, hcore, eri):
    """ Calculate the off-diagonal hamiltonian elements using the eris stored with
    4-fold symmetry
    square eri array (size n*(n-1)/2 square) (4-fold symmetry?)
    idet/jdet are a tuple of sets of occupied alpha beta orbitals differing by a
    single excitation
    """
    hij = 0.0
    hole, part, sign, spin, aocc, bocc = hole_part_sign_spin_occ_single_sets(idet, jdet)
    hij += hcore[part, hole]
    for si in (aocc, bocc)[spin]:
        hij += eri[idx4(part, hole, si, si)]
        hij -= eri[idx4(part, si, si, hole)]
    for si in (bocc, aocc)[spin]:
        hij += eri[idx4(part, hole, si, si)]
    hij *= sign
    return hij


def calc_hij_double_sets(idet, jdet, hcore, eri):
    """ Calculate the off-diagonal hamiltonian elements using the eris stored with
    4-fold symmetry
    square eri array (size n*(n-1)/2 square) (4-fold symmetry?)
    idet/jdet are a tuple of sets of occupied alpha beta orbitals differing by a
    double excitation
    """
    hij = 0.0
    h1, h2, p1, p2, sign, samespin = hole_part_sign_spin_double_sets(idet, jdet)
    hij += eri[idx4(p1, h1, p2, h2)]
    if samespin:
        hij -= eri[idx4(p1, h2, p2, h1)]
    hij *= sign
    return hij


def calc_hij_sets(idet, jdet, hcore, eri, nexc_ij=None):
    """ Calculate the off-diagonal hamiltonian elements using the eris stored with
    4-fold symmetry
    square eri array (size n*(n-1)/2 square) (4-fold symmetry?)
    idet/jdet are a tuple of sets of occupied alpha beta orbitals
    """
    if frozenset([idet, jdet]) in __hamdict:
        return __hamdict[frozenset([idet, jdet])]
    if nexc_ij is None:
        nexc_ij = n_excit_sets(idet, jdet)
    if nexc_ij == 0:
        __hamdict[frozenset([idet])] = calc_hii_sets(idet, hcore, eri)
    elif nexc_ij == 1:
        __hamdict[frozenset([idet, jdet])] = calc_hij_single_sets(idet, jdet, hcore, eri)
    elif nexc_ij == 2:
        __hamdict[frozenset([idet, jdet])] = calc_hij_double_sets(idet, jdet, hcore, eri)
    else:
        return 0.0
    return __hamdict[frozenset([idet, jdet])]


################################################################################
# FUNCTIONS FOR GENERATING CONNECTED DETERMINANTS
################################################################################


def get_excitations(det, norb, aexc, bexc):
    """
    usage: get_excitations(det, norb, aexc, bexc)
    returns a list of determinants connected to det with alpha(beta) excitation level aexc(bexc)
    norb is the number of spatial orbitals
    det should be a pair (2-tuple or list) or sets of occupied orbital indices)
    returned determinants will be 2-tuples of frozensets of occupied indices (alpha and beta)
    """

    aocc = det[0]
    bocc = det[1]
    orbs = frozenset(range(norb))
    avirt = orbs - aocc
    bvirt = orbs - bocc

    adets = []
    bdets = []

    if aexc > len(avirt) or aexc > len(aocc) or bexc > len(bvirt) or bexc > len(bocc):
        raise
    for iocc in itertools.combinations(aocc, aexc):
        for ivirt in itertools.combinations(avirt, aexc):
            adets.append(aocc - set(iocc) | set(ivirt))
    for iocc in itertools.combinations(bocc, bexc):
        for ivirt in itertools.combinations(bvirt, bexc):
            bdets.append(bocc - set(iocc) | set(ivirt))

    return [(i, j) for i in adets for j in bdets]


def gen_singles(det, norb):
    """ Generate determinants connected by a single excitation of an alpha or
    beta electron
    """
    return get_excitations(det, norb, 1, 0) + get_excitations(det, norb, 0, 1)


def gen_doubles(det, norb):
    """ Generate determinants connected by a double excitation of two alpha
    electrons, two beta electrons or 1 alpha and 1 beta electron
    """
    return get_excitations(det, norb, 2, 0) + get_excitations(det, norb, 0, 2)  + get_excitations(det, norb, 1, 1)


def gen_singles_doubles(det, norb):
    """Generate all single and double excitations connected to a determinant
    """
    return gen_singles(det, norb) + gen_doubles(det, norb)


################################################################################
# FUNCTIONS FOR GENERATING CONNECTED DETERMINANTS
################################################################################

def construct_ham_dict(coredetlist_sets, h1e, eri):
    """Where is my docstring?"""
    ham_dict = {}
    ndets = len(coredetlist_sets)
    for i in range(ndets):
        idet = coredetlist_sets[i]
        hii = calc_hii_sets(idet, h1e, eri)
        ham_dict[frozenset((idet))] = hii
        for j in range(i + 1, ndets):
            jdet = coredetlist_sets[j]
            nexc_ij = n_excit_sets(idet, jdet)
            if nexc_ij in (1, 2):
                if nexc_ij == 1:
                    hij = calc_hij_single_sets(idet, jdet, h1e, eri)
                else:
                    hij = calc_hij_double_sets(idet, jdet, h1e, eri)
                ham_dict[frozenset((idet, jdet))] = hij
    return ham_dict


def construct_hamiltonian(ndets, coredetlist_sets, h1e, eri):
    """Where is my docstring?"""
    hrow = []
    hcol = []
    hval = []
    for i in range(ndets):
        idet = coredetlist_sets[i]
        hii = calc_hii_sets(idet, h1e, eri)
        hrow.append(i)
        hcol.append(i)
        hval.append(hii)
        for j in range(i+1, ndets):
            jdet = coredetlist_sets[j]
            nexc_ij = n_excit_sets(idet, jdet)
            if nexc_ij in (1, 2):
                if nexc_ij == 1:
                    hij = calc_hij_single_sets(idet, jdet, h1e, eri)
                else:
                    hij = calc_hij_double_sets(idet, jdet, h1e, eri)
                hrow.append(i)
                hrow.append(j)
                hcol.append(j)
                hcol.append(i)
                hval.append(hij)
                hval.append(hij)
    return sp.sparse.csr_matrix((hval, (hrow, hcol)), shape=(ndets, ndets))

def getsmallham(dets, hamdict):
    """Whereis my docstring?"""
    hrow = []
    hcol = []
    hval = []
    ndets = len(dets)
    count = 0
    for i in range(ndets):
        idet = dets[i]
        hrow.append(i)
        hcol.append(i)
        hval.append(hamdict[frozenset((idet))])
        for j in range(i + 1, ndets):
            jdet = dets[j]
            if frozenset((idet, jdet)) in hamdict:
                hrow.append(i)
                hrow.append(j)
                hcol.append(j)
                hcol.append(i)
                hij = hamdict[frozenset((idet, jdet))]
                hval.append(hij)
                hval.append(hij)
            else:
                if n_excit_sets(idet, jdet) <= 2:
                    print(idet, jdet)
    return sp.sparse.csr_matrix((hval, (hrow, hcol)), shape=(ndets, ndets))


def getsmallhamslow(dets, hcore, eri):
    """Where is my docstring?"""
    hrow = []
    hcol = []
    hval = []
    ndets = len(dets)
    for i in range(ndets):
        idet = dets[i]
        hrow.append(i)
        hcol.append(i)
        hval.append(calc_hij_sets(idet, idet, hcore, eri))
        for j in range(i + 1, ndets):
            jdet = dets[j]
            hrow.append(i)
            hrow.append(j)
            hcol.append(j)
            hcol.append(i)
            hij = calc_hij_sets(idet, jdet, hcore, eri)
            hval.append(hij)
            hval.append(hij)
    return sp.sparse.csr_matrix((hval, (hrow, hcol)), shape=(ndets, ndets))


def populatehamdict(targetdetset, coreset, hamdict, h1e, eri):
    """Where is my docstring?"""
    update_dict = dict()
    for i in targetdetset:
        if i not in hamdict:
            hamdict[frozenset((i))] = calc_hii_sets(i, h1e, eri)
            for j in coreset:
                nexc_ij = n_excit_sets(i, j)
                if nexc_ij in (1, 2):
                    if nexc_ij == 1:
                        update_dict[frozenset([i, j])] = calc_hij_single_sets(i, j, h1e, eri)
                    else:
                        update_dict[frozenset([i, j])] = calc_hij_double_sets(i, j, h1e, eri)
    for i in coreset:
        if i not in hamdict:
            hamdict[frozenset((i))] = calc_hii_sets(i, h1e, eri)
            for j in coreset:
                nexc_ij = n_excit_sets(i, j)
                if nexc_ij in (1, 2):
                    if nexc_ij == 1:
                        update_dict[frozenset([i, j])] = calc_hij_single_sets(i, j, h1e, eri)
                    else:
                        update_dict[frozenset([i, j])] = calc_hij_double_sets(i, j, h1e, eri)
    return update_dict


########################################################### ASCI funcs

def asci(mol, cdets, tdets, convergence=1e-6, printroots=4, iter_min=0, visualize=False, preservedict=True):
    """Where is my docstring?"""

    if not preservedict:
        __hamdict={}
    print("PYCI")
    print("method: ASCI")
    #print("Paper: https://arxiv.org/abs/1603.02686")
    print("convergence: ", convergence)
    print("Core Space size: ", cdets)
    print("Target Space size: ", tdets)
    print("Number of eigenvalues: ", printroots)
    print("")
    Na, Nb = mol.nelec # get number of electrons
    E_nuc = mol.energy_nuc() # get nuclear repulsion energy
    nao = mol.nao_nr() # get number of spatial atomic orbitals which we assume is the same as the number of spatial molecular orbitals which is true when using RHF
    myhf = scf.RHF(mol) # create RHF object for molecule
    E_hf = myhf.kernel() # get Hartree-Fock energy
    mo_coefficients = myhf.mo_coeff # get MO coefficients
    h1e = reduce(np.dot, (mo_coefficients.T, myhf.get_hcore(), mo_coefficients)) # create matrix of 1 electron integrals
    print("transforming eris")
    eri = ao2mo.kernel(mol, mo_coefficients) # get 2 electron integrals
    hamdict = dict() # create dictionary of hamiltonian elements
    hfdet = (frozenset(range(Na)), frozenset(range(Nb))) # create Hartree-Fock determinant bit representation
    E_old = 0.0 # initalize previous iteration energy
    E_new = E_hf # initalize current iteration energy
    targetdetset = set() # initalize set of target determinants
    coreset = {hfdet} # initalize set of core determinants
    C = {hfdet : 1.0} # initalize dictionary of amplitudes as only HF det
    print("\nHartree-Fock Energy: ", E_hf)
    print("\nBeginning Iterations\n")
    it_num = 0
    while(np.abs(E_new - E_old) > convergence):
        it_num += 1
        E_old = E_new
        # step 1
        targetdetset = set() # clear set of target determinants
        for idet in coreset: # for each determinant in the core set of determinants...
            targetdetset |= set(gen_singles_doubles(idet, nao)) # ...generate all single and double excitations and add them to the target set
        A = dict.fromkeys(targetdetset, 0.0) # initalize a dictionary for the perturbed amplitudes of each determinant
        hamdict.update(populatehamdict(targetdetset, coreset, hamdict, h1e, eri)) # update the hamiltonian dictionary
        for idet in coreset: # for each determinant in the core set...
            for jdet in targetdetset: # ... and each in the target set...
                nexc_ij = n_excit_sets(idet, jdet) # find the exitation level between the two determinants
                if nexc_ij in (1, 2): # don't bother checking anything with a zero hamiltonian element (Slater Condon rules)
                    A[jdet] += hamdict[frozenset([idet, jdet])] * C[idet] # evaluate perturbed amplitudes (see paper eq 4)
        for idet in targetdetset: # for each determinant in the target set
            A[idet] /= (hamdict[frozenset((idet))] - E_old) # evaluate the denominator of eq 4 of the paper
        for idet in coreset: # for each determinant in the coreset...
            if idet in A: # if the determinant has a perturbed amplitude (which means its a )
                if abs(A[idet]) < abs(C[idet]): # replace with the biggest again
                    A[idet] = C[idet]
            else:
                A[idet] = C[idet]
        A_sorted = sorted(list(A.items()), key=lambda i: -abs(i[1]))
        A_truncated = A_sorted[:tdets]
        #print("Target Dets: ",len(A_truncated))
        A_dets = [i[0] for i in A_truncated]
        hamdict.update(populatehamdict(A_dets, A_dets, hamdict, h1e, eri))
        targetham = getsmallham(A_dets, hamdict)
        eig_vals,eig_vecs = sp.sparse.linalg.eigsh(targetham, k=2*printroots)
        eig_vals_sorted = np.sort(eig_vals)[:printroots]
        E_new = eig_vals_sorted[0]
        print("Iteration {:} Energy: ".format(it_num), E_new + E_nuc)
        # step 4
        amplitudes = eig_vecs[:, np.argsort(eig_vals)[0]]
        newdet = [i for i in zip(A_dets, amplitudes)]
        C = {}
        for i in sorted(newdet, key=lambda j: -abs(j[1]))[:cdets]:
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
    if visualize:
        visualize_sets(newdet, nao, Na, Nb, "ASCI")
    print("Completed ASCI!")

    return


############################################## HBCI functions


def heatbath(det, norb, hamdict, amplitudes, epsilon, h1e, eri, preservedict=True):
    """Where is my docstring?"""
    if not preservedict:
        __hamdict={}
    excitation_space = set()
    for i in det:
        excitation_space |= set(gen_singles(i, norb) + gen_doubles(i, norb))
    remove_set = set()
    for i in excitation_space:
        add = False
        for j in det:
            nexc_ij = n_excit_sets(i, j)
            if nexc_ij in (1, 2):
                if nexc_ij == 1:
                    h = calc_hij_single_sets(i, j, h1e, eri)
                    if abs(h * amplitudes[j]) >= epsilon:
                        add = True
                        hamdict[frozenset([i, j])] = h
                else:
                    h = calc_hij_double_sets(i, j, h1e, eri)
                    if abs(h * amplitudes[j]) >= epsilon:
                        add = True
                        hamdict[frozenset([i, j])] = h
        if add == False:
            remove_set.add(i)
    return excitation_space - remove_set, hamdict


def hbci(mol, epsilon=0.01, convergence=0.01, printroots=4, visualize=False, preservedict=True):
    """Where is my docstring?"""
    print(preservedict)
    if not preservedict:
        __hamdict={}
    print("PYCI")
    print("method: HBCI")
    print("Paper: https://arxiv.org/abs/1606.07453")
    print("convergence: ", convergence)
    print("Epsilon: ", epsilon)
    print("Number of eigenvalues: ", printroots)
    print("")
    Na, Nb = mol.nelec # nelec is a tuple with (N_alpha, N_beta)
    E_nuc = mol.energy_nuc()
    nao = mol.nao_nr()
    myhf = scf.RHF(mol)
    E_hf = myhf.kernel()
    mo_coefficients = myhf.mo_coeff
    h1e = reduce(np.dot, (mo_coefficients.T, myhf.get_hcore(), mo_coefficients))
    eri = ao2mo.kernel(mol, mo_coefficients)
    num_orbs = 2 * nao
    num_occ = mol.nelectron
    num_virt = num_orbs - num_occ
    E_old = 0.0
    E_new = E_hf
    hfdet = (frozenset(range(Na)), frozenset(range(Nb)))
    oldselecteddetset= {hfdet}
    C = {hfdet : 1.0}
    print("Hartree-Fock Energy: ", E_hf)
    print("")
    it_num = 1
    Converged = False
    hamdict = dict()
    while not Converged:
        newselecteddetset = set()
        newselecteddetset, hamdict_additions = heatbath(oldselecteddetset, nao, hamdict, C, epsilon, h1e, eri)
        hamdict.update(hamdict_additions)
        hamdict.update(populatehamdict(newselecteddetset, oldselecteddetset, hamdict, h1e, eri))
        newselecteddetset |= oldselecteddetset
        hamdict.update(populatehamdict(newselecteddetset, newselecteddetset, hamdict, h1e, eri))
        selectedham = getsmallham(list(newselecteddetset), hamdict)
        eig_vals,eig_vecs = sp.sparse.linalg.eigsh(selectedham, k=2 * printroots)
        eig_vals_sorted = np.sort(eig_vals)[:printroots]
        E_new = eig_vals_sorted[0]
        amplitudes = eig_vecs[:, np.argsort(eig_vals)[0]]
        newdet = [i for i in zip(newselecteddetset, amplitudes)]
        C = {}
        for i in sorted(newdet, key=lambda j: -abs(j[1])):
            C[i[0]] = i[1]
        print("Iteration {:} Energy: ".format(it_num), E_new + E_nuc)
        amount_added = abs(len(newselecteddetset) - len(oldselecteddetset))
        if amount_added < np.ceil(convergence * len(oldselecteddetset)):
            Converged = True
        else:
            print("Iteration {:} space growth %: ".format(it_num), amount_added)
        oldselecteddetset |= newselecteddetset
        it_num += 1
        E_old = E_new
        print("Selected Space size: ", len(oldselecteddetset))
        print("")
    print("first {:} pyci eigvals".format(printroots))
    for i in (eig_vals_sorted + E_nuc):
        print(i)
    if visualize:
        visualize_sets(newdet, nao, Na, Nb, "HBCI")
    print("Completed HBCI!")

    return


########################################################### CISD funcs


def cisd(mol, printroots=4, visualize=False, preservedict=True):
    """Where is my docstring?"""
    if not preservedict:
        __hamdict={}
    print("PYCI")
    print("method: CISD")
    print("Number of eigenvalues: ", printroots)
    print("")
    Na,Nb = mol.nelec # get number of electrons
    E_nuc = mol.energy_nuc() # get nuclear repulsion energy
    nao = mol.nao_nr() # get number of spatial atomic orbitals which we assume is the same as the number of spatial molecular orbitals which is true when using RHF
    myhf = scf.RHF(mol) # create RHF object for molecule
    E_hf = myhf.kernel() # get Hartree-Fock energy
    mo_coefficients = myhf.mo_coeff # get MO coefficients
    h1e = reduce(np.dot, (mo_coefficients.T, myhf.get_hcore(), mo_coefficients)) # create matrix of 1 electron integrals
    print("transforming eris")
    eri = ao2mo.kernel(mol, mo_coefficients) # get 2 electron integrals
    hamdict = dict() # create dictionary of hamiltonian elements
    hfdet = (frozenset(range(Na)),frozenset(range(Nb))) # create Hartree-Fock determinant bit representation
    targetdetset = set() # initalize set of target determinants
    coreset = {hfdet} # initalize set of core determinants
    print("\nHartree-Fock Energy: ", E_hf)
    for idet in coreset: # for each determinant in the core set of determinants...
        targetdetset |= set(gen_singles_doubles(idet,nao)) # ...generate all single and double excitations and add them to the target set
    targetdetset |= coreset # create a set from the union of the target and core determinants
    hamdict.update(populatehamdict(targetdetset,targetdetset,hamdict,h1e,eri)) # update the dictionary of hamiltonian elements with those of the target determinants
    targetham = getsmallham(list(targetdetset),hamdict) # construct the hamiltonian in the target space
    eig_vals,eig_vecs = sp.sparse.linalg.eigsh(targetham,k=2*printroots) # diagonalize the hamiltonian yielding the eigenvalues and eigenvectors
    eig_vals_sorted = np.sort(eig_vals)[:printroots] # sort the eigenvalues
    print("")
    print("first {:} pyci eigvals".format(printroots))
    for i in (eig_vals_sorted + E_nuc):
        print(i)
    print(np.shape(eig_vecs))
    if visualize:
        newdet = [i for i in zip(list(targetdetset),eig_vecs[:,np.argsort(eig_vals)[0]])]
        visualize_sets(newdet,nao,Na,Nb,"CISD")
    print("Completed CISD!")

###########################################################FCI funcs

def fci(mol,printroots=4,visualize=False,preservedict=True):
    if not preservedict:
        __hamdict={}
    print("PYCI")
    print("method: FCI")
    print("Number of eigenvalues: ",printroots)
    print("")
    Na,Nb = mol.nelec #nelec is a tuple with (N_alpha, N_beta)
    E_nuc = mol.energy_nuc()
    nao = mol.nao_nr()
    myhf = scf.RHF(mol)
    E_hf = myhf.kernel()
    mo_coefficients = myhf.mo_coeff
    h1e = reduce(np.dot, (mo_coefficients.T, myhf.get_hcore(), mo_coefficients))
    print("transforming eris")
    eri = ao2mo.kernel(mol, mo_coefficients)
    print("generating all determinants")
    fulldetlist_sets=gen_dets_sets(nao,Na,Nb)
    ndets=len(fulldetlist_sets)
    print("constructing full Hamiltonian")
    full_hamiltonian = construct_hamiltonian(ndets,fulldetlist_sets,h1e,eri)
    eig_vals,eig_vecs = sp.sparse.linalg.eigsh(full_hamiltonian,k=2*printroots)
    eig_vals_sorted = np.sort(eig_vals)[:printroots]
    print("")
    print("first {:} pyci eigvals".format(printroots))
    for i in (eig_vals_sorted + E_nuc):
        print(i)
    if visualize:
        print(len(fulldetlist_sets),len(eig_vecs[:,np.argsort(eig_vals)[0]]))
        newdet = [i for i in zip(list(fulldetlist_sets),eig_vecs[:,np.argsort(eig_vals)[0]])]
        visualize_sets(newdet,nao,Na,Nb,"FCI")
    print("Completed FCI!")

########################################################### ACI

def aci(mol,sigma = 100,gamma = 0.0001,convergence = 1e-10,printroots=4,iter_min=0,visualize=False):
    print("PYCI")
    print("method: ACI")
    print("Paper: http://dx.doi.org/10.1063/1.4948308")
    print("convergence: ", convergence)
    print("Sigma: ",sigma)
    print("Gamma: ",gamma)
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
        #step 2
        targetdetset=set()
        for idet in coreset:
            targetdetset |= set(gen_singles_doubles(idet,nao))
        A = dict.fromkeys(targetdetset, 0.0)
        hamdict.update(populatehamdict(targetdetset , coreset,hamdict,h1e,eri))
        #equation 5
        for jdet in targetdetset:
            V = 0
            for idet in coreset:
                nexc_ij = n_excit_sets(idet,jdet)
                if nexc_ij in (1,2): # don't bother checking anything with a zero hamiltonian element
                    try:
                        V += hamdict[frozenset([idet,jdet])] * C[idet]
                    except:
                        V += hamdict[frozenset([jdet,idet])] * C[idet]
            delta  = (hamdict[frozenset((jdet))] - E_new)/2.0
            A[jdet] = delta - np.sqrt((delta**2.0) + V**2.0)
        #step 4
        A_sorted = sorted(list(A.items()),key=lambda i: -abs(i[1]))
        #cumulative energy error eq 6
        count = 0
        err = 0.0
        while (abs(err)<=sigma and count < len(A_sorted)):
            err += A_sorted[count][1]
            count +=1
        #step 5
        A_truncated = A_sorted[:count]
        A_dets = [i[0] for i in A_truncated]
        A_dets += list(coreset)
        A_dets = list(set(A_dets))
        print("Q space size: ",len(A_dets))
        hamdict.update(populatehamdict(A_dets,A_dets,hamdict,h1e,eri))
        targetham = getsmallham(A_dets,hamdict)
        eig_vals,eig_vecs = sp.sparse.linalg.eigsh(targetham,k=2*printroots)
        eig_vals_sorted = np.sort(eig_vals)[:printroots]
        E_new = eig_vals_sorted[0]
        print("Iteration {:} Energy: ".format(it_num), E_new + E_nuc)
        #step 4
        amplitudes = eig_vecs[:,np.argsort(eig_vals)[0]]
        newdet = [i for i in zip(A_dets,amplitudes)]
        C = {}
        #eq 10
        sorted_newdet = sorted(newdet,key=lambda j: -abs(j[1]))
        err = 0
        count = 0
        while abs(err) <= 1-(gamma*sigma):
            err += sorted_newdet[count][1]**2
            C[sorted_newdet[count][0]] = sorted_newdet[count][1]
            count +=1
        print(count)
        if sorted(newdet,key=lambda j: -abs(j[1]))[0][0] != hfdet:
            print("Biggest Contributor is NOT HF det ", sorted(newdet,key=lambda j: -abs(j[1]))[0])
        coreset = set(C.keys())
        print("")
    #import pickle
    #with open('ACI_DETS.txt', 'wb') as handle:
        #pickle.dump(newdet, handle)
    if visualize:
        visualize_sets(newdet,nao,Na,Nb,"ACI")
    print("first {:} pyci eigvals".format(printroots))
    for i in (eig_vals_sorted + E_nuc):
        print(i)
    print("Completed ACI!")

########################################################### visualization

if __name__ == "__main__":
    print("\nPYCI utils file. This file was not meant to be run independently.")
