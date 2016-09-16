import scipy as sp
import scipy as sp
import scipy.linalg as spla
import numpy as np
from functools import reduce
import pyscf
from pyscf import gto, scf, ao2mo, fci, mp, ao2mo
def myump2(mf):
    # As UHF objects, mo_energy, mo_occ, mo_coeff are two-item lists
    # (the first item for alpha spin, the second for beta spin).
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    mo_coeff = mf.mo_coeff
    o = np.hstack((mo_coeff[0][:,mo_occ[0]>0] ,mo_coeff[1][:,mo_occ[1]>0]))
    v = np.hstack((mo_coeff[0][:,mo_occ[0]==0],mo_coeff[1][:,mo_occ[1]==0]))
    eo = np.hstack((mo_energy[0][mo_occ[0]>0] ,mo_energy[1][mo_occ[1]>0]))
    ev = np.hstack((mo_energy[0][mo_occ[0]==0],mo_energy[1][mo_occ[1]==0]))
    no = o.shape[1]
    nv = v.shape[1]
    noa = sum(mo_occ[0]>0)
    nva = sum(mo_occ[0]==0)
    eri = ao2mo.general(mf.mol, (o,v,o,v)).reshape(no,nv,no,nv)
    eri[:noa,nva:] = eri[noa:,:nva] = eri[:,:,:noa,nva:] = eri[:,:,noa:,:nva] = 0
    g = eri - eri.transpose(0,3,2,1)
    eov = eo.reshape(-1,1) - ev.reshape(-1)
    de = 1/(eov.reshape(-1,1) + eov.reshape(-1)).reshape(g.shape)
    emp2 = .25 * np.einsum('iajb,iajb,iajb->', g, g, de)
    return emp2
mol = gto.M(
    atom = [['O', (0.000000000000,  -0.143225816552,   0.000000000000)],
            ['H', (1.638036840407,   1.136548822547,  -0.000000000000)],
            ['H', (-1.638036840407,   1.136548822547,  -0.000000000000)]],
    basis = 'STO-3G',
    verbose = 1,
    unit='b'
)
myhf = scf.RHF(mol)
E = myhf.kernel()
ijkl = ao2mo.incore.full(pyscf.scf._vhf.int2e_sph(mol._atm, mol._bas, mol._env), myhf.mo_coeff, compact=False)
print E
print myump2(myhf)
