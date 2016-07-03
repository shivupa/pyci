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

