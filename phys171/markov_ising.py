##---------------------------------------------------------------------
## PROGRAM : markov_ising.py
## PURPOSE : implement SMAC algorithm 5.7 (markov-ising).
##           (Metropolis algorithm). With plotting of configurations 
##           (using pylab)
## LANGUAGE: Python 2.5
##---------------------------------------------------------------------
from random import uniform as ran, randint, choice
from math import exp
from pylab import hist, figure, savefig, show, plot, axis
##---------------------------------------------------------------------
## Sample geometry
##---------------------------------------------------------------------
def square_neighbors(L):
    N = L * L
    site_dic = {}
    x_y_dic = {}
    for j in range(N):
        row = j // L
        column = j - row * L
        site_dic[(row, column)] = j
        x_y_dic[j] = (row, column)
    nbr = []
    for j in range(N):
        row, column = x_y_dic[j]
        right_nbr = site_dic[row, (column + 1) % L]
        up_nbr = site_dic[(row + 1) % L, column]
        left_nbr = site_dic[row, (column - 1 + L) % L]
        down_nbr = site_dic[(row - 1 + L) % L, column]
        nbr.append((right_nbr, up_nbr, left_nbr, down_nbr))
    nbr = tuple(nbr)
    return nbr, site_dic, x_y_dic

#
# Program starts here
#
L = 32
N = L * L
S = [choice([-1, 1]) for k in range(N)]
beta = 0.42
nbr, site_dic, x_y_dic = square_neighbors(L)
for i_sweep in range(100):
    for iteri in range(N):
        k = randint(0, N - 1)
        h = sum(S[nbr[k][j]] for j in range(4))
        Delta_E = 2. * h * S[k]
        Upsilon = exp(-beta * Delta_E)
        if ran(0., 1.) < Upsilon: S[k] = -S[k]
figure()
x_plus = []
y_plus = []
x_minus = []
y_minus = []
for i in range(N):
    x, y = x_y_dic[i]
    if S[i] == 1:
        x_plus.append(x)
        y_plus.append(y)
    else:
        x_minus.append(x)
        y_minus.append(y)
axis('scaled')
axis([-0.5, L - 0.5, -0.5, L - .5])
plot(x_plus, y_plus, 'r+', markersize=10)
plot(x_minus, y_minus, 'bx', markersize=10)
savefig('test2.eps')
show()