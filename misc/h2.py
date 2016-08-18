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
     H    0.000000000000     0.000000000000     0.000000000000
     H    0.000000000000     0.000000000000     0.100000000000
  '''
mol.basis = 'aug-cc-pvtz'
mol.unit='b'
mol.verbose=7
mol.build()
#
# Pass 1, generate all HOMOs with external field
#
myhf = scf.RHF(mol)
myhf2 = scf.RHF(mol).apply(scf.addons.remove_linear_dep_)
e_hf=myhf.kernel()
e_hf2=myhf2.kernel()
s = mol.intor('cint1e_ovlp_sph')
s2 = mol.intor('cint1e_ovlp_cart')
cmo = myhf.mo_coeff
cmo2=myhf2.mo_coeff
eigs = np.linalg.eigh(s)
