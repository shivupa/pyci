import numpy as np

from pyci.tests.runner import runner


def test_h2o_dz():

    from pyci.examples.h2o_dz import mol

    energies = runner(mol)

    energies_gamess = [
        -75.0129802245,
        -74.7364625517,
        -74.6886742417,
        -74.6531877287
    ]

    for k in energies:
        assert len(energies[k]) == 4

    for rootidx, (e_pyscf, e_gamess, e_pyci) in enumerate(zip(energies['fci_pyscf'],
                                                              energies_gamess,
                                                              energies['fci_pyci'])):
        # This is a basic floating point equality test to some threshold.
        assert abs(e_pyscf - e_pyci) < 1.0e-12
        assert abs(e_pyscf - e_gamess) < 1.0e-7
        # This is more characters, but also more general. Decide on which one
        # once we restructure things/know py.test better.
        # assert np.testing.assert_almost_equal(e_pyscf, e_pyci, decimal=12)
        # assert np.testing.assert_almost_equal(e_pyscf, e_gamess, decimal=7)

    return


def test_cn_sto3g():

    from pyci.examples.cn_sto3g import mol

    # Disable this for now since ACI takes forever.

    # energies = runner(mol)

    # for k in energies:
    #     assert len(energies[k]) == 4

    # for rootidx, (e_pyscf, e_pyci) in enumerate(zip(energies['fci_pyscf'],
    #                                                 energies['fci_pyci'])):
    #     assert abs(e_pyscf - e_pyci) < 1.0e-12
