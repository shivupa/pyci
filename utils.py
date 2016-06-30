
def idx2(i,j):
    if i>j:
        return int(i*(i+1)/2+j)
    else:
        return int(j*(j+1)/2+i)

def idx4(i,j,k,l):
    return (idx2(i,j),idx2(k,l))

def n_excit(idet,jdet):
    if idet==jdet:
        return 0
    return (bin(int(idet,2)^int(jdet,2)).count('1'))/2

def hamweight(strdet):
    return strdet.count('1')

#slower implementations
#--------------------------------------------
#def ham1(strdet):
#    return sum(map(int,list(strdet)))
#def ham2(strdet):
#    return sum(i=="1" for i in strdet)
#def ham3(strdet):
#    s=0
#    for i in strdet:
#        if i=="1":
#            s+=1
#    return s
#def ham4(strdet):
#    return sum([1 for i in strdet if i=="1"])
#--------------------------------------------
