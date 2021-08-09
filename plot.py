import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import plottingdata
from matplotlib.ticker import FormatStrFormatter
matplotlib.rcParams.update({'font.size': 25})


Full = plottingdata.Full
NoMorph =plottingdata.NoMorph
NoCaus = plottingdata.NoCaus
x = np.array([ 3.5, 3.25,3,2.75, 2.5,2.25, 2, 1.75, 1.5, 1.25, 1, 0.75,0.5])
m = 0
n = 100
x0 = 3
xn = 15
s = 2
a = 0.5

fig, (ax1, ax2) = plt.subplots(2, 1)
for i in range(m,n):
    ax1.plot(x[x0:xn], Full[i][26][x0:xn], label='goal', color='#E6C200', linewidth = s, alpha = a)
    ax2.plot(x[x0:xn], np.add(Full[i][5][x0:xn], Full[i][18][x0:xn]), label='intinf', color='black', linewidth=s,
             alpha=a)
ax1.set(ylabel=r"$P_1(g_1)$", xlabel='sensor length')
ax2.set(ylabel=r"$\Phi_{TIF}$", xlabel='sensor length')
plt.xlabel("sensor length")
plt.show()

fig, (ax1, ax2) = plt.subplots(2, 1)
for i in range(m,n):
    ax1.plot(x[x0:xn], NoMorph[i][26][x0:xn], label='goal', color='#E6C200', linewidth = s, alpha = a)
    ax2.plot(x[x0:xn], np.add(NoMorph[i][5][x0:xn], NoMorph[i][18][x0:xn]), label='intinf', color='black', linewidth=s,
             alpha=a)
ax1.set(ylabel=r"$P_2(g_1)$", xlabel='sensor length')
ax2.set(ylabel=r"$\Phi_{TIF}$", xlabel='sensor length')
plt.xlabel("sensor length")
plt.show()

fig, (ax1, ax2) = plt.subplots(2, 1)
for i in range(m,n):
    ax1.plot(x[0:10], NoCaus[i][26][0:10], label='goal', color='#E6C200', linewidth = s, alpha = a)
    ax2.plot(x[x0:xn], np.add(NoCaus[i][5][0:10], NoCaus[i][18][0:10]), label='intinf', color='black', linewidth=s, alpha=a)
ax1.set(ylabel=r"$P_3(g_1)$", xlabel='sensor length')
ax2.set(ylabel=r"$\Phi_{TIF}$", xlabel='sensor length')
plt.xlabel("sensor length")
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2)
for i in range(m,n):
     ax1.plot(x[x0:xn], np.add(Full[i][2][x0:xn], Full[i][15][x0:xn]), label='intinf', color='#325283',  linewidth = s, alpha = a)
     ax2.plot(x[x0:xn], np.add(Full[i][9][x0:xn], Full[i][22][x0:xn]), label='goal', color='#228b22', linewidth = s, alpha = a)
ax1.set(ylabel=r"$\Phi_{T}$", xlabel = 'sensor length' )
ax2.set(ylabel=r"$\Psi_{S}$", xlabel = 'sensor length' )

plt.show()


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
for i in range(m,n):
    ax2.plot(x[x0:xn], np.add(Full[i][6][x0:xn], Full[i][19][x0:xn]), label='policy', color='#325283', linewidth = s, alpha = a)
    ax1.plot(x[x0:xn], np.add(Full[i][7][x0:xn], Full[i][20][x0:xn]), label='strategy', color='#630615',  linewidth = s, alpha = a)
    ax3.plot(x[x0:xn], np.add(Full[i][1][x0:xn], Full[i][14][x0:xn]), label='goal', color='#630615', linewidth=s, alpha=a)
    ax4.plot(x[x0:xn], (np.add(Full[i][10][x0:xn], Full[i][23][x0:xn])-2)*(-2), label='goal', color='#228b22',  linewidth = s, alpha = a)

ax2.set(ylabel=r"$\Psi_{C}$", xlabel = 'sensor length' )
ax1.set(ylabel=r"$\Psi_{SI}$", xlabel = 'sensor length' )
ax3.set(ylabel=r"$\Psi_{R}$", xlabel = 'sensor length' )
ax4.set(ylabel=r"$\Psi_{A}$", xlabel = 'sensor length' )
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2)
s = 2
for i in range(m,n):
     ax1.plot(x[x0:xn], np.add(NoMorph[i][2][x0:xn], NoMorph[i][15][x0:xn]), label='intinf', color='#325283',  linewidth = s, alpha = a)
     ax2.plot(x[x0:xn], np.add(NoMorph[i][9][x0:xn], NoMorph[i][22][x0:xn]), label='goal', color='#228b22', linewidth = s, alpha = a)
ax1.set(ylabel=r"$\Phi_{T}$", xlabel = 'sensor length' )
ax2.set(ylabel=r"$\Psi_{S}$", xlabel = 'sensor length' )

plt.show()

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)

for i in range(m,n):
    ax2.plot(x[x0:xn], np.add(NoMorph[i][6][x0:xn], NoMorph[i][19][x0:xn]), label='policy', color='#325283', linewidth = s, alpha = a)
    ax1.plot(x[x0:xn], np.add(NoMorph[i][7][x0:xn], NoMorph[i][20][x0:xn]), label='strategy', color='#630615',  linewidth = s, alpha = a)
    ax3.plot(x[x0:xn], np.subtract(Full[i][26][x0:xn], NoMorph[i][26][x0:xn]), color ='#CFAE00' , alpha=a)
    ax4.plot(x[x0:xn], (np.add(NoMorph[i][10][x0:xn], NoMorph[i][23][x0:xn])-2)*(-2), label='goal', color='#228b22',  linewidth = s, alpha = a)
    ax5.plot(x[x0:xn], np.add(NoMorph[i][12][x0:xn], NoMorph[i][25][x0:xn]), label='policy', color='#325283', linewidth = s, alpha = a)
    ax6.plot(x[x0:xn], np.add(NoMorph[i][11][x0:xn], NoMorph[i][24][x0:xn]), label='policy', color='#228b22', linewidth=s, alpha=a)
    ax3.plot(x[x0:xn], np.zeros(10), color = 'black', linewidth = 0.5*s)
ax2.set(ylabel=r"$\Psi_{C}$", xlabel = 'Sensor Length' )
ax1.set(ylabel=r"$\Psi_{SI}$", xlabel = 'Sensor Length' )
ax4.set(ylabel=r"$\Psi_{A}$", xlabel = 'Sensor Length' )
ax5.set(ylabel=r"$\Psi_{PI}$", xlabel = 'Sensor Length' )
ax6.set(ylabel=r"$\Psi_{Syn}$", xlabel = 'Sensor Length' )
ax3.set(ylabel=r"$P_1 (g_1) - P_2 (g_1)$", xlabel = 'Sensor Length' )
#ax3.set(yscale = 'symlog')
# ax1.set(yscale = 'log')
# ax4.set(yscale = 'log')
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2)
for i in range(m,n):
     ax1.plot(x[x0:xn], np.add(NoCaus[i][2][0:10], NoCaus[i][15][0:10]), label='intinf', color='#325283',  linewidth = s, alpha = a)
     ax2.plot(x[x0:xn], np.add(NoCaus[i][9][0:10], NoCaus[i][22][0:10]), label='goal', color='#228b22', linewidth = s, alpha = a)
ax1.set(ylabel=r"$\Phi_{T}$", xlabel = 'Sensor Length' )
ax2.set(ylabel=r"$\Psi_{S}$", xlabel = 'Sensor Length' )

plt.show()


fig, ((ax1, ax2), (ax3, ax4), (ax5,ax6)) = plt.subplots(3, 2)

for i in range(m,n):
    ax1.plot(x[x0:xn], np.add(NoCaus[i][7][0:10], NoCaus[i][20][0:10]), label='goal', color='#630615', linewidth = s, alpha = a)
    ax3.plot(x[x0:xn], np.add(NoCaus[i][1][0:10], NoCaus[i][14][0:10]), label='strategy', color='#630615',  linewidth = s, alpha = a)
    ax2.plot(x[x0:xn], np.subtract(Full[i][26][x0:xn], NoCaus[i][26][0:10]), color ='#CFAE00' , alpha=a)
    ax4.plot(x[x0:xn], np.add(NoCaus[i][10][0:10], NoCaus[i][23][0:10]), label='intinf', color='#325283', linewidth=s, alpha=a)
    ax5.plot(x[x0:xn], np.add(NoCaus[i][12][0:10], NoCaus[i][25][0:10]), label='policy', color='#325283', linewidth=s, alpha=a)
    ax6.plot(x[x0:xn], np.add(NoCaus[i][11][0:10], NoCaus[i][24][0:10]), label='policy', color='#228b22', linewidth=s, alpha=a)

# ax4.plot(x[x0:xn], (np.add(NoCaus[i][10][x0:xn], NoCaus[i][23][x0:xn])-2)*(-2), label='goal', color='#228b22',  linewidth = s, alpha = a)

ax1.set(ylabel=r"$\Psi_{SI}$", xlabel = 'Sensor Length' )
ax4.set(ylabel=r"$\Psi_{A}$", xlabel = 'Sensor Length' )
ax3.set(ylabel=r"$\Psi_{R}$", xlabel = 'Sensor Length' )
ax5.set(ylabel=r"$\Psi_{PI}$", xlabel = 'Sensor Length' )
ax6.set(ylabel=r"$\Psi_{Syn}$", xlabel = 'Sensor Length' )
ax2.set(ylabel=r"$P_1 (g_1) - P_3 (g_1)$", xlabel = 'Sensor Length' )
plt.show()


fig = plt.figure(figsize=(3, 3))
ax1 = plt.subplot2grid((15, 15), (0, 0), colspan=4, rowspan=5)
ax2 = plt.subplot2grid((15, 15), (0, 5), colspan=4, rowspan=5)
ax3 = plt.subplot2grid((15, 15), (0, 10), colspan=4, rowspan=5)
ax4 = plt.subplot2grid((15, 15), (5, 0), colspan=4, rowspan=5)
ax5 = plt.subplot2grid((15, 15), (5, 5), colspan=4, rowspan=5)
ax6 = plt.subplot2grid((15, 15), (6, 10), colspan=4, rowspan=4)
ax7 = plt.subplot2grid((15, 15), (10, 0), colspan=4, rowspan=5)
ax8 = plt.subplot2grid((15, 15), (10, 5), colspan=4, rowspan=5)
ax9 = plt.subplot2grid((15, 15), (11, 10), colspan=4, rowspan=4)
ax1.set_title('Controller')
ax2.set_title('Morphological Computation')
ax3.set_title('Predictive Information')
ax9.set_title('Goal Difference')

for i in range(m,n):
    ax1.plot(x[x0:xn], np.add(NoMorph[i][7][x0:xn], NoMorph[i][20][x0:xn]), label='strategy', color='#325283', linewidth = s, alpha = a)
    ax4.plot(x[x0:xn], np.add(NoMorph[i][6][x0:xn], NoMorph[i][19][x0:xn]), label='policy', color='#325283', linewidth = s, alpha = a)
    ax1.plot(x[x0:xn], np.add(Full[i][7][x0:xn], Full[i][20][x0:xn]), label='strategy', color='black', linewidth=s, alpha=a)
    ax4.plot(x[x0:xn], np.add(Full[i][6][x0:xn], Full[i][19][x0:xn]), label='policy', color='black', linewidth=s, alpha=a)

    ax7.plot(x[x0:xn], np.add(NoMorph[i][2][x0:xn], NoMorph[i][15][x0:xn]), label='intinf', color='#6C90C7',linewidth = s,alpha = a)
    ax7.plot(x[x0:xn], np.add(Full[i][2][x0:xn], Full[i][15][x0:xn]), label='intinf', color='black', linewidth=s, alpha=a)

    ax3.plot(x[x0:xn], np.add(NoMorph[i][12][x0:xn], NoMorph[i][25][x0:xn]), label='goal', color='#228b22', linewidth = s, alpha = a)
    ax3.plot(x[x0:xn], np.add(NoMorph[i][0][x0:xn], NoMorph[i][13][x0:xn]), label='goal', color='#145314', linewidth=s, alpha = a)

    ax3.plot(x[x0:xn], np.add(Full[i][12][x0:xn], Full[i][25][x0:xn]), label='goal', color='black', linewidth=s, alpha=a)
    ax3.plot(x[x0:xn], np.add(Full[i][0][x0:xn], Full[i][13][x0:xn]), label='goal', color='black', linewidth=s, alpha=a)

    ax5.plot(x[x0:xn], np.add(NoMorph[i][11][x0:xn], NoMorph[i][24][x0:xn]), label='goal', color='#228b22', linewidth = s, alpha = a)
    ax8.plot(x[x0:xn], np.add(NoMorph[i][9][x0:xn], NoMorph[i][22][x0:xn]), label='goal', color='#228b22', linewidth = s, alpha = a)
    ax8.plot(x[x0:xn], np.add(Full[i][9][x0:xn], Full[i][22][x0:xn]), label='goal', color='black', linewidth=s, alpha=a)
    ax5.plot(x[x0:xn], np.add(Full[i][11][x0:xn], Full[i][24][x0:xn]), label='goal', color='black', linewidth=s, alpha=a)
    ax2.plot(x[x0:xn],(np.add(Full[i][10][x0:xn], Full[i][23][x0:xn]) -2) *(-2), label='goal', color='black', linewidth=s, alpha=a)

    ax2.plot(x[x0:xn], (np.add(NoMorph[i][10][x0:xn], NoMorph[i][23][x0:xn]) -2)*(-2), label='goal', color='#228b22', linewidth = s, alpha = a)
   # ax5.plot(x[x0:xn], np.add(NoMorph[i][10][x0:xn], NoMorph[i][10][x0:xn]), label='goal', color='#228b22', linewidth = s)

ax1.set(ylabel=r"$\Psi_{SI}$")
ax4.set(ylabel=r"$\Psi_{C}$" )
ax7.set(ylabel=r"$\Psi_{M}, \Phi_{T} $", xlabel='Sensor Length' )
ax7.legend([r"$\Psi_{M}$", r"$\Phi_{T}$"], loc ="upper right")
#ax9.set(ylabel=r"$\Psi_{SA}- \Phi_{PI} $" )

ax5.set(ylabel=r"$\Psi_{Syn}$")
ax2.set(ylabel=r"$\Psi_{A}$" )
ax8.set(ylabel=r"$\Psi_{S}$", xlabel='Sensor Length' )
ax6.set(ylabel=r"$\Psi_{SA}- \Psi_{S}$")
ax9.set(ylabel=r"$P_1(g_1) - P_3(g_1)$", xlabel='Sensor Length')
ax3.set(ylabel=r"$\Psi_{PI}, \Psi_{SA}$" )

fig = plt.figure(figsize=(3, 3))
ax1 = plt.subplot2grid((15, 15), (0, 0), colspan=4, rowspan=5)
ax2 = plt.subplot2grid((15, 15), (0, 5), colspan=4, rowspan=5)
ax3 = plt.subplot2grid((15, 15), (0, 10), colspan=4, rowspan=5)
ax4 = plt.subplot2grid((15, 15), (5, 0), colspan=4, rowspan=5)
ax5 = plt.subplot2grid((15, 15), (7, 5), colspan=4, rowspan=5)
ax7 = plt.subplot2grid((15, 15), (10, 0), colspan=4, rowspan=5)
ax8 = plt.subplot2grid((15, 15), (7, 10), colspan=4, rowspan=5)
ax2.set_title('Reactive Control')
ax1.set_title('Morphological Computation')
ax5.set_title('Predictive Information')
ax3.set_title('Goal Difference')

for i in range(m,n):
    ax2.plot(x[x0:xn], np.add(Full[i][1][x0:xn], Full[i][14][x0:xn]), label='goal', color='black', linewidth=s, alpha=a)
    ax2.plot(x[x0:xn], np.add(NoCaus[i][1][x0:xn], NoCaus[i][14][x0:xn]), label='goal', color='#630615',  linewidth = s, alpha = a)
    ax2.plot(x[x0:xn], np.add(NoCaus[i][8][x0:xn], NoCaus[i][21][x0:xn]), label='goal', color='#A50A23',  linewidth = s, alpha = a)
    ax2.plot(x[x0:xn], np.add(Full[i][8][x0:xn], Full[i][21][x0:xn]), label='goal', color='black', linewidth=s, alpha=a)

    ax4.plot(x[x0:xn], np.add(NoCaus[i][11][x0:xn], NoCaus[i][24][x0:xn]), label='goal', color='#228b22', linewidth = s, alpha = a)
    ax7.plot(x[x0:xn], np.add(NoCaus[i][9][x0:xn], NoCaus[i][22][x0:xn]), label='goal', color='#228b22',  linewidth = s, alpha = a)
    ax1.plot(x[x0:xn], np.add(NoCaus[i][10][x0:xn], NoCaus[i][23][x0:xn]), label='goal', color='#228b22',  linewidth = s, alpha = a)

    ax4.plot(x[x0:xn], np.add(Full[i][11][x0:xn], Full[i][24][x0:xn]), label='goal', color='black', linewidth=s, alpha=a)
    ax7.plot(x[x0:xn], np.add(Full[i][9][x0:xn], Full[i][22][x0:xn]), label='goal', color='black', linewidth=s, alpha=a)
    ax1.plot(x[x0:xn], np.add(Full[i][10][x0:xn], Full[i][23][x0:xn]), label='goal', color='black', linewidth=s, alpha=a)

    ax5.plot(x[x0:xn], np.add(NoCaus[i][0][x0:xn], NoCaus[i][13][x0:xn]), label='goal', color='#145314', linewidth=s, alpha = a)
    ax5.plot(x[x0:xn], np.add(Full[i][0][x0:xn], Full[i][13][x0:xn]), label='goal', color='black', linewidth=s, alpha=a)
    ax3.plot(x[x0:xn], np.subtract(Full[i][26][x0:xn], NoCaus[i][26][x0:xn]), alpha=a, linewidth = s, color = '#CFAE00' )

    ax8.plot(x[x0:xn], np.subtract(np.add(NoCaus[i][0][x0:xn], NoCaus[i][13][x0:xn]), np.add(NoCaus[i][9][x0:xn], NoCaus[i][22][x0:xn])), label='goal', color='#145314', linewidth=s)

ax1.set(ylabel=r"$\Psi_{A}$" )
ax7.set(ylabel=r"$\Psi_{S}$", xlabel = 'Sensor Length' )
ax4.set(ylabel=r"$\Psi_{Syn}$")
ax5.set(ylabel=r"$\Psi_{SA}, \Psi_{PI}$",   xlabel = 'Sensor Length' )
ax8.set(ylabel=r"$\Psi_{SA} - \Psi_{S}$", xlabel='Sensor Length' )
ax2.set(ylabel=r"$\Psi_{R},\Psi_{MSI}$" )
ax2.legend([r"$\Psi_{R}$", r"$\Psi_{MSI}$"], loc ="lower right")
ax5.legend([r"$\Psi_{PI}$", r"$\Psi_{SA}$"], loc ="lower right")
ax3.set(ylabel=r"$P_1(g_1) - P_2(g_1)$")