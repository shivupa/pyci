#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
This example has two parts.  The first part applies oscillated electric field
by modifying the 1-electron Hamiltonian.  The second part generate input
script for Jmol to plot the HOMOs.  Running jmol xxxx.spt can output 50
image files of HOMOs under different electric field.
'''

import numpy as np
from pyscf import gto, scf, tools

mol = gto.Mole() # Benzene
mol.atom = '''
     C    0.000000000000     1.398696930758     0.000000000000
     C    0.000000000000    -1.398696930758     0.000000000000
     C    1.211265339156     0.699329968382     0.000000000000
     C    1.211265339156    -0.699329968382     0.000000000000
     C   -1.211265339156     0.699329968382     0.000000000000
     C   -1.211265339156    -0.699329968382     0.000000000000
     H    0.000000000000     2.491406946734     0.000000000000
     H    0.000000000000    -2.491406946734     0.000000000000
     H    2.157597486829     1.245660462400     0.000000000000
     H    2.157597486829    -1.245660462400     0.000000000000
     H   -2.157597486829     1.245660462400     0.000000000000
     H   -2.157597486829    -1.245660462400     0.000000000000
  '''
mol.basis = 'aug-cc-pvtz'
mol.build()
mol.unit='b'
#
# Pass 1, generate all HOMOs with external field
#
myhf = scf.RHF(mol)
e_hf=myhf.kernel()
s = mol.intor('cint1e_ovlp_sph')
s2 = mol.intor('cint1e_ovlp_cart')
cmo = myhf.mo_coeff

eigs = np.linalg.eigh(s)
