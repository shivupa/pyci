import numpy as np

import matplotlib.pyplot as plt

from pyci.methods.utils import gen_dets_sets

def visualize_sets(a, nao, Na, Nb, name):

    fulldetset = gen_dets_sets(nao, Na, Nb)
    full = np.array(list(fulldetset))
    occupied_dets = np.array(a)[:, 0]
    occupied_amps = np.array(a)[:, 1]
    occ_plot = np.zeros(len(full))
    for i in range(len(occupied_dets)):
        for j in range(len(fulldetset)):
            if (occupied_dets[i][0] == fulldetset[j][0] and occupied_dets[i][1] == fulldetset[j][1]):
                occ_plot[j] = occupied_amps[i]
    plt.rc('font', family='serif', size=24)
    fig = plt.figure(facecolor='white', figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20
    np.savetxt("{}_plot1.txt".format(name), np.arange(len(occ_plot)))
    np.savetxt("{}_plot2.txt".format(name), occ_plot)
    plt.bar(np.arange(len(occ_plot)), np.abs(occ_plot), color='k', align='center')
    ax.set_yscale('log')
    ax.xaxis.set_tick_params(pad=7, width=1.5, length=10)
    ax.yaxis.set_tick_params(pad=7, width=1.5, length=10)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
    plt.xlabel('Determinant Index')
    plt.ylabel('Amplitude')
    plt.title('Wavefunction plot', y=1.08)
    fig.savefig('{}.svg'.format(name), bbox_inches='tight')

    return

if __name__ == "__main__":

    pass
