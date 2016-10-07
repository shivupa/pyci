from __future__ import print_function

import time
import numpy as np

import pyscf
import pyscf.fci

from pyci.methods.utils import fci
from pyci.methods.utils import cisd
from pyci.methods.utils import asci
from pyci.methods.utils import hbci
from pyci.methods.utils import aci

np.set_printoptions(precision=4, suppress=True)

# pylint: disable=invalid-name


def runner(mol):

    ###################################

    print("FCI from PYSCF")
    start = time.time()
    myhf = pyscf.scf.RHF(mol)
    E = myhf.kernel()
    c = myhf.mo_coeff
    cisolver = pyscf.fci.FCI(mol, c)
    energies_fci_pyscf = cisolver.kernel(nroots=4)[0] + mol.energy_nuc()
    print(energies_fci_pyscf)
    for e in energies_fci_pyscf:
        print(e)
    end = time.time()
    print("RUN TIME PYSCF: (h:m:s)")
    m, s = divmod(end - start, 60)
    h, m = divmod(m, 60)
    print("%d:%02d:%02d" % (h, m, s))

    ###################################

    print(79 * "~")
    print(30 * " " + "FCI")
    print(79 * "~")
    start = time.time()
    energies_fci_pyci = fci(mol, printroots=4, visualize=False)
    end = time.time()
    print("RUN TIME FCI: (h:m:s)")
    m, s = divmod(end - start, 60)
    h, m = divmod(m, 60)
    print("%d:%02d:%02d" % (h, m, s))

    ###################################

    print(79 * "~")
    print(30 * " " + "CISD")
    print(79 * "~")
    start = time.time()
    energies_cisd = cisd(mol, printroots=4, visualize=False)
    end = time.time()
    print("RUN TIME CISD: (h:m:s)")
    m, s = divmod(end - start, 60)
    h, m = divmod(m, 60)
    print("%d:%02d:%02d" % (h, m, s))

    ###################################

    print(79 * "~")
    print(30 * " " + "ASCI")
    print(79 * "~")
    start = time.time()
    energies_asci = asci(mol, cdets=50, tdets=100, convergence=10e-9, printroots=4, iter_min=10, visualize=False)
    end = time.time()
    print("RUN TIME ASCI: (h:m:s)")
    m, s = divmod(end - start, 60)
    h, m = divmod(m, 60)
    print("%d:%02d:%02d" % (h, m, s))

    ###################################

    print(79 * "~")
    print(30 * " " + "HBCI")
    print(79 * "~")
    start = time.time()
    energies_hbci = hbci(mol, epsilon=0.01, convergence=0.01, printroots=4, visualize=False)
    end = time.time()
    print("RUN TIME HBCI: (h:m:s)")
    m, s = divmod(end - start, 60)
    h, m = divmod(m, 60)
    print("%d:%02d:%02d" % (h, m, s))

    ###################################

    print(79 * "~")
    print(30 * " " + "ACI")
    print(79 * "~")
    start = time.time()
    energies_aci = aci(mol, sigma=100, gamma=0.0001, convergence=1e-10, printroots=4, iter_min=0, visualize=False)
    end = time.time()
    print("RUN TIME ACI: (h:m:s)")
    m, s = divmod(end - start, 60)
    h, m = divmod(m, 60)
    print("%d:%02d:%02d" % (h, m, s))

    ###################################

    energies = dict()

    energies['fci_pyscf'] = energies_fci_pyscf
    energies['fci_pyci'] = energies_fci_pyci
    energies['cisd'] = energies_cisd
    energies['asci'] = energies_asci
    energies['hbci'] = energies_hbci
    energies['aci'] = energies_aci

    return energies


if __name__ == '__main__':
    pass
