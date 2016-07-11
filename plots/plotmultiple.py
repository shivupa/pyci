import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
xvec = np.loadtxt('ACI_plot1.txt')
ACI = np.loadtxt('ACI_plot2.txt')
ASCI = np.loadtxt('ASCI_plot2.txt')
FCI = np.loadtxt('FCI_plot2.txt')
CISD = np.loadtxt('CISD_plot2.txt')
HBCI = np.loadtxt('HBCI_plot2.txt')
plt.rc('font', family='serif',size = 24)
fig = plt.figure(facecolor='white',figsize=(10,10))
ax = fig.add_subplot(111)
ax.xaxis.labelpad = 20
ax.yaxis.labelpad = 20
colors = ['darkorchid','dodgerblue','#00b300','#f8d022','darkorange','tomato']
colors.reverse()
plt.bar(xvec,ACI, color = colors[0],edgecolor = colors[0],align='center',alpha = 0.4)
plt.bar(xvec,ASCI, color = colors[1],edgecolor = colors[1],align='center',alpha = 0.4)
plt.bar(xvec,FCI, color = 'k',align='center',alpha = 0.4)
plt.bar(xvec,CISD, color = colors[2],edgecolor = colors[2],align='center',alpha = 0.4)
plt.bar(xvec,HBCI, color = colors[3],edgecolor = colors[3],align='center',alpha = 0.4)

ax.set_yscale('log')
ax.xaxis.set_tick_params(pad=7,width=1.5,length=10)
ax.yaxis.set_tick_params(pad = 7,width=1.5,length=10)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.5)
plt.xlabel(r'Determinant Index')
plt.ylabel(r'Amplitude')
plt.title(r'Wavefunction plot',y=1.08)
fig.tight_layout()
fig.savefig('Alltogether.svg')
