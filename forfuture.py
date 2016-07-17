def n_excit_spin(idet,jdet,spin):
    """get the hamming weight of a bitwise xor on two determinants.
    This will show how the number of orbitals in which they have different
    occupations"""
    if idet[spin]==jdet[spin]:
        return 0
    return (bin(int(idet[spin],2)^int(jdet[spin],2)).count('1'))/2
def n_excit(idet,jdet):
    """get the hamming weight of a bitwise xor on two determinants.
    This will show how the number of orbitals in which they have different
    occupations"""
    if idet==jdet:
        return 0
    aexc = n_excit_spin(idet,jdet,0)
    bexc = n_excit_spin(idet,jdet,1)
    return aexc+bexc
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
