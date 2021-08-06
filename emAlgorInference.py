import numpy as np
from SampledData import inference05new  as infer05
from SampledData import inference075new as infer075
from SampledData import inference1new as infer1
from SampledData import inference15new as infer15
from SampledData import inference2new as infer2
from SampledData import inference125new as infer125
from SampledData import inference175new as infer175
from SampledData import inference225 as infer225
from SampledData import inference25 as infer25
from SampledData import inference275new as infer275
from scipy import sparse
from math import log2
import random

def emAlg(initial,internalnodes, iterations, toll, mode, base):
    results = np.zeros((27,10))
    # the different sensor length
    inference = [infer275, infer25, infer225, infer2, infer175, infer15, infer125, infer1, infer075, infer05]
    for i in range(10):
        pwsa = np.zeros(2**4)
        world = inference[i].world + 1
        world = world/np.sum(world)
        for k in range(2**6):
            pwsa[k//4] = pwsa[k//4] + world[k]
        initialsa = np.zeros(2**4)
        for k in range(2**(2*internalnodes +9)):
            initialsa[(k//(2**(internalnodes + 4)))%4 +4*((k//(2**(2*internalnodes +6)))%4)] = initialsa[(k//(2**(internalnodes + 4)))%4 +4*((k//(2**(2*internalnodes +6)))%4)] + initial[k]
        for k in range(2**(2*internalnodes +9)):
            initial[k] = initial[k]/initialsa[(k//(2**(internalnodes + 4)))%4 +4*((k//(2**(2*internalnodes +6)))%4)] * pwsa[(k//(2**(internalnodes + 4)))%4 +4*((k//(2**(2*internalnodes +6)))%4)]
        initial = initial/ np.sum(initial)
        p, newworld2, world, pinitial, goal, pgcas = emAlgit(initial,internalnodes, iterations, toll, inference[i],mode, base)
        newp = np.zeros( 2**(8 +2*internalnodes) )
        for j in range(2**(9+2*internalnodes )):
            newp[j%(2**(8 +2*internalnodes))] = newp[j%(2**(8 +2*internalnodes))] + p[j]
        pmarg = np.zeros(2**(internalnodes + 4))
        for j in range(2**(8+2*internalnodes)):
            pmarg[j//(2**(internalnodes +4))] = pmarg[j//(2**(internalnodes +4))] + newp[j]

        a, b, c, d, e, f, g, h, l, m, n, o, q ,\
        a1, b1, c1, d1, e1, f1, g1, h1, l1, m1, n1, o1, q1= calculateMeasures(newp,pmarg, newworld2, world,0, internalnodes)
        results[0][i] = a
        results[1][i] = b
        results[2][i] = c
        results[3][i] = d
        results[4][i] = e
        results[5][i] = f
        results[6][i] = g
        results[7][i] = h
        results[8][i] = l
        results[9][i] = m
        results[10][i] = n
        results[11][i] = o
        results[12][i] = q

        results[13][i] = a1
        results[14][i] = b1
        results[15][i] = c1
        results[16][i] = d1
        results[17][i] = e1
        results[18][i] = f1
        results[19][i] = g1
        results[20][i] = h1
        results[21][i] = l1
        results[22][i] = m1
        results[23][i] = n1
        results[24][i] = o1
        results[25][i] = q1
        results[26][i] = goal
        print(results)
    return results

def emAlgit(initial,internalnodes, iterations, toll, infer, mode, base):
    # load the sampled data and calculate the resulting conditional distributions
    # pgcas = P(G | S_t+2, A_t+2,  S_t+1, A_t+1, S_t, A_t)
    # world = P(S_t+1  | S_t, A_t)
    goal = infer.goal +1
    goal = goal/ np.sum(goal)
    pgcas = np.zeros(2 ** 13)
    pas = np.zeros(2 ** 12)
    goal = sparse.csr_matrix(goal)
    newworld2 = infer.world +1
    newworld2 = newworld2/ np.sum(newworld2)
    psalast = np.zeros(16)
    world = np.copy(newworld2)
    for i in range(2 ** 13):
        pas[i % (2 ** 12)] = pas[i % (2 ** 12)] + goal.__getitem__((0, i))
    for i in range(2 ** 13):
        if pas[i % (2 ** 12)] != 0:
            pgcas[i] = goal.__getitem__((0, i)) / pas[i % (2 ** 12)]
        else:
            pgcas[i] = 0.5
    for i in range(64):
        psalast[i // 4] = psalast[i // 4] + world[i]

    for i in range(64):
        if psalast[i // 4] != 0:
            world[i] = world[i] / psalast[i // 4]
        else:
            world[i] = 0.25
    pinitial = np.zeros(2**(internalnodes +4))
    # calculate the fixed P(S_t | A_t) and P(C_t | S_t, A_t), P(A_t) from the initial distribution
    psa = np.zeros(16)
    pa = np.zeros(4)
    for i in range(2 **(9 +2*internalnodes)):
        pinitial[(i // (2**(4+internalnodes))) % (2**(internalnodes+4))] = pinitial[(i // (2**(4+internalnodes))) % (2**(internalnodes+4))] + initial[i]
        psa[(i//(2**(internalnodes+4)))%4 + 4 * (i // (2 ** (2*internalnodes + 6)) % 4)] = psa[(i//(2**(internalnodes+4)))%4 + 4 * (i // (2 ** (2*internalnodes + 6)) % 4)] +initial[i]
        pa[(i // (2 ** (internalnodes + 4))) % 4] = pa[(i // (2 ** (internalnodes + 4))) % 4] + initial[i]
    psca = np.copy(psa)
    for i in range(16):
        psca[i] = psca[i]/ pa[i%4]
    for i in range(2**(internalnodes +4)):
        pinitial[i] = pinitial[i]/psa[i%4+ 4*(i//(2**(internalnodes+2)))]
    p = factorizing(initial,pinitial, psca,pa, world, pgcas, internalnodes, mode, base)
    goal3 = 0
    goaltoll = True
    k = 0
    toll2 = 1
    while goaltoll:
            if toll2 < toll and k > iterations:
                goaltoll = False
            p ,goal2 = conditioning(p, internalnodes)
            psa = np.zeros(16)
            pa = np.zeros(4)
            pinitial = np.zeros(2 ** (internalnodes + 4))
            for i in range(2 ** (9 + 2 * internalnodes)):
                pinitial[(i // (2 ** (4 + internalnodes))) % (2 ** (internalnodes + 4))] = pinitial[(i // ( 2 ** (4 + internalnodes))) % (2 ** (internalnodes + 4))] + p[i]
                psa[(i // (2 ** (internalnodes + 4))) % 4 + 4 * (i // (2 ** (2 * internalnodes + 6)) % 4)] = psa[(i // (2 ** (internalnodes + 4))) % 4 + 4 * (i // (2 ** (2 * internalnodes + 6)) % 4)] + p[i]
                pa[(i // (2 ** (internalnodes + 4))) % 4] = pa[(i // (2 ** (internalnodes + 4))) % 4] + p[i]
            for i in range(2 ** (internalnodes + 4)):
                pinitial[i] = pinitial[i] / psa[i % 4 + 4 * (i // (2 ** (internalnodes + 2)))]
            p = factorizing(p,pinitial, psca,pa, world, pgcas, internalnodes,mode, base)
            toll2 = abs(goal2-goal3)
            goal3 = goal2
            k = k+1
    return p, newworld2, world,pinitial, goal2, pgcas

#e-projection by conditioning on the goal
def conditioning(p, internalnodes):
    goal = np.zeros(2)
    for i in range(2**(9+2*internalnodes)):
        goal[i//(2**(8+2*internalnodes))] = goal[i//(2**(8+2*internalnodes))] +p[i]
    nextp = np.zeros(2**(9+2*internalnodes))
    for i in range(2**(9+2*internalnodes)):
        if i > 2**(8+2*internalnodes):
            nextp[i] = p[i]/goal[1]
        else:
            nextp[i] = 0
    nextp = nextp/(np.sum(nextp))
    return nextp, goal[1]

# the m-projection, factorizing according to the architecture of the agents
def factorizing(p,pinitial, psca, pa1, world, pgcas, internalnodes, mode, base):
    if mode == 0:
        pa = np.zeros((2,2))
        psc = np.zeros(2**(2+internalnodes))
        pacsc = np.zeros((2, 2**(3+internalnodes)))
        pccsc = np.zeros((internalnodes, 2**(3+internalnodes)))
        newp = np.zeros(2**(9+ 2*internalnodes))
        pgcsasca = np.zeros(2**(9+internalnodes))
        pnextsteps = np.zeros(16)
        for i in range(2**(9+2*internalnodes)):
            psc[(i // (2**(internalnodes +6)))%(2**(internalnodes+2))] = psc[(i // (2**(internalnodes +6)))%(2**(internalnodes+2))] + p[i]
            for j in range(2):
                pacsc[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2**(internalnodes +6)))%(2**(internalnodes+2)))] = pacsc[j][(i//(2 ** (1 - j))) % 2 + 2 * ((i // (2**(internalnodes +6)))%(2**(internalnodes+2)))] + p[i]
                pa[j][(i // (2 ** (1 - j))) % 2] = pa[j][(i // (2 ** (1 - j))) % 2] + p[i]
            for j in range(internalnodes):
                pccsc[j][(i // (2 ** (internalnodes +1 - j))) % 2 + 2 * ((i // (2**(internalnodes +6)))%(2**(internalnodes+2)))] = pccsc[j][(i // (2 ** (internalnodes +1 - j))) % 2 + 2 * ((i // (2**(internalnodes +6)))%(2**(internalnodes+2)))] + p[i]
        summ = 0
        for i in range( 2**(3+internalnodes)):
            for j in range(2):
                if psc[i//2] !=0 :
                    pacsc[j][i] = pacsc[j][i] / psc[i // 2]
                else:
                    pacsc[j][i] = 0.5
       # #print("pacsc", pacsc)
        for i in range(2**(9+internalnodes)):
            for j in range(16):
                if(base):
                    pnextsteps[j] = pa[0][(j // 2) % 2] * pa[1][j % 2 ]  * pgcas[(i // (2 ** (8 + internalnodes))) * (2 ** 12) + ((i // (2 ** (internalnodes + 2))) % (2 ** 6)) * (2 ** 6) + (i % 4) * (2 ** 4) + j] * world[(j // 4) % 4 + 4 * (i % 4) + 16 * ((i // (2 ** (internalnodes + 2))) % 4)]
                else:
                    pnextsteps[j] = pacsc[0][(j // 2) % 2 + 2 * ((i // 4) % (2 ** (internalnodes +2)))] * pacsc[1][j % 2 + 2 * ((i // 4) % (2 ** (internalnodes +2)))] * pgcas[(i//(2**(8+internalnodes)))*(2**12)  + ((i // (2 ** (internalnodes +2)))% (2**6)) * (2 ** 6)  + (i % 4) * (2 ** 4) + j] * world[(j // 4) % 4 + 4 * (i % 4) + 16 * ((i // (2 ** (internalnodes +2))) % 4)]
            pgcsasca[i] = pgcsasca[i] + np.sum(pnextsteps)
            summ = summ + np.sum(pnextsteps)

        for i in range(2**(3+internalnodes)):
            for j in range(internalnodes):
                if psc[i // 2] != 0:
                    pccsc[j][i] = pccsc[j][i]/psc[i//2]
                else:
                    pccsc[j][i] = 0.5
        for i in range(2**(9+ 2*internalnodes)):
            newp[i] = pinitial[(i//(2**(internalnodes +4)))%(2**(internalnodes+4))] * psca[(i//(2**(internalnodes +4)))%4 + 4*((i//(2**(2*internalnodes +6)))%4)] * pa1[(i//(2**(internalnodes+4)))%4] * world[(i//(2**(internalnodes +2)))%(2**4) + (2**4)*((i//(2**(2*internalnodes +6)))%4)]
            for j in range(internalnodes):
                newp[i] = newp[i] * pccsc[j][(i // (2 ** (internalnodes +1 - j))) % 2 + 2 * ((i // (2**(internalnodes +6)))%(2**(internalnodes+2)))]
            for j in range(2):
                if(base):
                    newp[i] = newp[i] * pa[j][(i // (2 ** (1 - j))) % 2]
                else:
                    newp[i] = newp[i] * pacsc[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2**(internalnodes +6)))%(2**(internalnodes+2)))]
            newp[i] = newp[i] * pgcsasca[(i//(2**(8+2*internalnodes)))*(2**(8+internalnodes))  + ((i // (2 ** (6+2*internalnodes)))%4) * (2 ** (6+internalnodes)) + (i % (2 ** (6+internalnodes)))]
    if mode == 1:
        pa = np.zeros((2, 2))
        psc = np.zeros(2 ** (2 + internalnodes))
        pacc = np.zeros((2, 2 ** (internalnodes + 1)))
        pc = np.zeros(2 ** internalnodes)
        pccsc = np.zeros((internalnodes, 2 ** (3 + internalnodes)))
        newp = np.zeros(2 ** (9 + 2 * internalnodes))
        pgcsasca = np.zeros(2 ** (9 + internalnodes))
        pnextsteps = np.zeros(16)
        for i in range(2 ** (9 + 2 * internalnodes)):
            psc[(i // (2 ** (internalnodes + 6))) % (2 ** (internalnodes + 2))] = psc[(i // (2 ** (internalnodes + 6))) % (2 ** (internalnodes + 2))] + p[i]
            pc[(i // (2 ** (internalnodes + 6))) % (2 ** internalnodes)] = pc[(i // (2 ** (internalnodes + 6))) % ( 2 ** internalnodes)] + p[i]
            for j in range(2):
                pacc[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (internalnodes + 6))) % (2 ** (internalnodes)))] = pacc[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (internalnodes + 6))) % (2 ** (internalnodes)))] +  p[i]
                pa[j][(i // (2 ** (1 - j))) % 2] = pa[j][(i // (2 ** (1 - j))) % 2] + p[i]
            for j in range(internalnodes):
                pccsc[j][(i // (2 ** (internalnodes + 1 - j))) % 2 + 2 * ( (i // (2 ** (internalnodes + 6))) % (2 ** (internalnodes + 2)))] = pccsc[j][(i // ( 2 ** (internalnodes + 1 - j))) % 2 + 2 * ((i // (2 ** (internalnodes + 6))) % ( 2 ** (internalnodes + 2)))] + p[i]
        for i in range(2 ** (internalnodes + 1)):
            for j in range(2):
                if pc[i // 2] != 0:
                    pacc[j][i] = pacc[j][i] / pc[i // 2]
                else:
                    pacc[j][i] = 0.5

        for i in range(2 ** (3 + internalnodes)):
            for j in range(internalnodes):
                if psc[i // 2] != 0:
                    pccsc[j][i] = pccsc[j][i] / psc[i // 2]
                else:
                    pccsc[j][i] = 0.5
        for i in range(2 ** (9 + internalnodes)):
            for j in range(16):
                if (base):
                    pnextsteps[j] = pa[0][(j // 2) % 2] * pa[1][j % 2] * pgcas[(i // (2 ** (8 + internalnodes))) * (2 ** 12) + ((i // (2 ** (internalnodes + 2))) % (2 ** 6)) * (2 ** 6) + (i % 4) * (2 ** 4) + j] * world[(j // 4) % 4 + 4 * (i % 4) + 16 * ((i // (2 ** (internalnodes + 2))) % 4)]
                else:
                    pnextsteps[j] = pacc[0][(j // 2) % 2 + 2 * ((i // 4) % (2 ** internalnodes))] * pacc[1][j % 2 + 2 * ((i // 4) % (2 ** internalnodes))] * pgcas[(i // (2 ** (8 + internalnodes))) * (2 ** 12) + ((i // (2 ** (internalnodes + 2))) % (2 ** 6)) * (2 ** 6) + ( i % 4) * (2 ** 4) + j] * world[ (j // 4) % 4 + 4 * (i % 4) + 16 * ((i // (2 ** (internalnodes + 2))) % 4)]
            pgcsasca[i] = pgcsasca[i] + np.sum(pnextsteps)
        for i in range(2 ** (9 + 2 * internalnodes)):
            newp[i] = pinitial[(i // (2 ** (internalnodes + 4))) % (2 ** (internalnodes + 4))] * psca[ (i // (2 ** (internalnodes + 4))) % 4 + 4 * ((i // (2 ** (2 * internalnodes + 6))) % 4)] * pa1[(i // (2 ** (internalnodes + 4))) % 4] * world[(i // (2 ** (internalnodes + 2))) % (2 ** 4) + (2 ** 4) * ((i // (2 ** (2 * internalnodes + 6))) % 4)]
            for j in range(internalnodes):
                newp[i] = newp[i] * pccsc[j][(i // (2 ** (internalnodes + 1 - j))) % 2 + 2 * ((i // (2 ** (internalnodes + 6))) % (2 ** (internalnodes + 2)))]
            for j in range(2):
                if (base):
                    newp[i] = newp[i] * pa[j][(i // (2 ** (1 - j))) % 2]
                else:
                    newp[i] = newp[i] * pacc[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (internalnodes + 6))) % (2 ** internalnodes))]
            newp[i] = newp[i] * pgcsasca[(i // (2 ** (8 + 2 * internalnodes))) * (2 ** (8 + internalnodes)) + ((i // (2 ** (6 + 2 * internalnodes))) % 4) * (2 ** (6 + internalnodes)) + ( i % (2 ** (6 + internalnodes)))]
    if mode ==2:
        pa = np.zeros((2, 2))
        pacsc = np.zeros((2, 2 ** (3 + internalnodes)))
        pccsc = np.zeros((internalnodes, 2 ** (3 + internalnodes)))
        pacs = np.zeros((2, 2 ** (3)))
        pc = np.zeros(2 ** internalnodes)
        pccs = np.zeros((internalnodes, 8))
        pccscj = np.zeros((internalnodes, 16))
        ps = np.zeros(4)
        psc = np.zeros(2 ** (2 + internalnodes))
        pscj = np.zeros((internalnodes, 8))
        newp = np.zeros(2 ** (9 + 2 * internalnodes))
        pgcsasca = np.zeros(2 ** (9 + internalnodes))
        pnextsteps = np.zeros(16)
        pc2 = np.zeros((internalnodes, 2))
        for i in range(2 ** (9 + 2 * internalnodes)):
            ps[(i // (2 ** (6 + 2 * internalnodes))) % 4] = ps[(i // (2 ** (6 + 2 * internalnodes))) % 4] + p[i]
            pc[(i // (2 ** 2)) % (2 ** internalnodes)] = pc[(i // (2 ** 2)) % (2 ** internalnodes)] + p[i]
            psc[(i // (2 ** (internalnodes + 6))) % (2 ** (internalnodes + 2))] = psc[(i // ( 2 ** (internalnodes + 6))) % (2 ** (internalnodes + 2))] + p[i]
            for j in range(2):
                pacsc[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (internalnodes + 6))) % (2 ** (internalnodes + 2)))] = pacsc[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (internalnodes + 6))) % (2 ** (internalnodes + 2)))] + p[i]
                pa[j][(i // (2 ** (1 - j))) % 2] = pa[j][(i // (2 ** (1 - j))) % 2] + p[i]
                pacs[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (2 * internalnodes + 6))) % (2 ** (2)))] = pacs[j][( i // ( 2 ** ( 1 - j))) % 2 + 2 * ( (i // ( 2 ** ( 2 * internalnodes + 6))) % (2 ** (2)))] +  p[i]
            for j in range(internalnodes):
                pccsc[j][(i // (2 ** (internalnodes + 1 - j))) % 2 + 2 * ((i // (2 ** (internalnodes + 6))) % (2 ** (internalnodes + 2)))] = pccsc[j][(i // (2 ** (internalnodes + 1 - j))) % 2 + 2 * ((i // (2 ** (internalnodes + 6))) % (2 ** (internalnodes + 2)))] + p[i]
                pc2[j][(i // (2 ** (internalnodes + 1 - j))) % 2] = pc2[j][(i // (2 ** (internalnodes + 1 - j))) % 2] + p[i]
                pccs[j][(i // (2 ** (internalnodes + 1 - j))) % 2 + 2 * ((i // (2 ** (2 * internalnodes + 6))) % 4)] = pccs[j][(i // (2 ** (internalnodes + 1 - j))) % 2 + 2 * ((i // (2 ** (2 * internalnodes + 6))) % 4)] + p[i]
                pccscj[j][(i // (2 ** (internalnodes + 1 - j))) % 2 + 2 * ( (i // (2 ** (2 * internalnodes + 5 - j))) % 2) + 4 * ((i // (2 ** (6 + 2 * internalnodes))) % 4)] = pccscj[j][(i // (2 ** (internalnodes + 1 - j))) % 2 + 2 * ((i // (2 ** (2 * internalnodes + 5 - j))) % 2) + 4 * ((i // (2 ** (6 + 2 * internalnodes))) % 4)] + p[i]
                pscj[j][ (i // (2 ** (2 * internalnodes + 5 - j))) % 2 + 2 * ((i // (2 ** (6 + 2 * internalnodes))) % 4)] = pscj[j][ (i // (2 ** (2 * internalnodes + 5 - j))) % 2 + 2 * ((i // (2 ** (6 + 2 * internalnodes))) % 4)] + p[i]
        for i in range(2 ** (3 + internalnodes)):
            for j in range(2):
                if psc[i // 2] != 0:
                    pacsc[j][i] = pacsc[j][i] / psc[i // 2]
                else:
                    pacsc[j][i] = 0.5
        for i in range(2 ** (3 + internalnodes)):
            for j in range(internalnodes):
                if psc[i // 2] != 0:
                    pccsc[j][i] = pccsc[j][i] / psc[i // 2]
                else:
                    pccsc[j][i] = 0.5
        for i in range(8):
            if ps[i // 2] != 0:
                for j in range(internalnodes):
                    pccs[j][i] = pccs[j][i] / ps[i // 2]
                for j in range(2):
                    pacs[j][i] = pacs[j][i] / ps[i // 2]
            else:
                for j in range(internalnodes):
                    pccs[j][i] = 0.5
                for j in range(2):
                    pacs[j][i] = 0.5
        for i in range(2 ** (9 + internalnodes)):
            for j in range(16):
                if (base):
                    pnextsteps[j] = pa[0][(j // 2) % 2] * pa[1][j % 2] * pgcas[(i // (2 ** (8 + internalnodes))) * (2 ** 12) + ( (i // (2 ** (internalnodes + 2))) % (2 ** 6)) * (2 ** 6) + (i % 4) * (2 ** 4) + j] * world[(j // 4) % 4 + 4 * (i % 4) + 16 * ((i // (2 ** (internalnodes + 2))) % 4)]
                else:
                    pnextsteps[j] = pacs[0][(j // 2) % 2 + 2 * ((i // (2 ** (internalnodes + 2))) % (2 ** (2)))] * pacs[1][j % 2 + 2 * ((i // (2 ** (internalnodes + 2))) % (2 ** (2)))] * pgcas[ (i // (2 ** (8 + internalnodes))) * (2 ** 12) + ((i // (2 ** (internalnodes + 2))) % (2 ** 6)) * (2 ** 6) + (i % 4) * (2 ** 4) + j] * world[(j // 4) % 4 + 4 * (i % 4) + 16 * ((i // (2 ** (internalnodes + 2))) % 4)]
            pgcsasca[i] = pgcsasca[i] + np.sum(pnextsteps)
        for i in range(2 ** (9 + 2 * internalnodes)):
            newp[i] = pinitial[(i // (2 ** (internalnodes + 4))) % (2 ** (internalnodes + 4))] * psca[(i // (2 ** (internalnodes + 4))) % 4 + 4 * ((i // (2 ** (2 * internalnodes + 6))) % 4)] * pa1[ (i // (2 ** (internalnodes + 4))) % 4] * world[ (i // (2 ** (internalnodes + 2))) % (2 ** 4) + (2 ** 4) * ((i // (2 ** (2 * internalnodes + 6))) % 4)]
            for j in range(internalnodes):
                newp[i] = newp[i] * pccsc[j][(i // (2 ** (internalnodes + 1 - j))) % 2 + 2 * ((i // (2 ** (internalnodes + 6))) % (2 ** (internalnodes + 2)))]

            for j in range(2):
                if (base):
                    newp[i] = newp[i] * pa[j][(i // (2 ** (1 - j))) % 2]
                else:
                    newp[i] = newp[i] * pacs[j][ (i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (2 * internalnodes + 6))) % (2 ** (2)))]
            newp[i] = newp[i] * pgcsasca[(i // (2 ** (8 + 2 * internalnodes))) * (2 ** (8 + internalnodes)) + ( (i // (2 ** (6 + 2 * internalnodes))) % 4) * (2 ** (6 + internalnodes)) + (i % (2 ** (6 + internalnodes)))]

    return newp

#calculating the measures
def calculateMeasures(p, psca1,  worldneu, world, mode, internalnodes):
    p0 = np.copy(p)
    psw = np.zeros(4)
    paw = np.zeros(4)
    psca = np.zeros(2 ** 4)
    pscs = np.zeros(2 ** 4)
    newworld = np.zeros(4)
    pas = np.zeros(16)
    for i in range(64):
        newworld[i % 4] = newworld[i % 4] + worldneu[i]
        pas[i // 4] = pas[i // 4] + worldneu[i]
        psw[(i // 16)] = psw[(i // 16)] + worldneu[i]
        paw[(i // 4) % 4] = paw[(i // 4) % 4] + worldneu[i]
        psca[i % 16] = psca[i % 16] + worldneu[i]
        pscs[(i % 4) + 4 * (i // 16)] = pscs[(i % 4) + 4 * (i // 16)] + worldneu[i]

    for i in range(16):
        if paw[i // 4] != 0:
            psca[i] = psca[i] / paw[i // 4]
        else:
            psca[i] = 1 / 4
        if psw[i // 4] != 0:
            pscs[i] = pscs[i] / psw[i // 4]
        else:
            pscs[i] = 1 / 4
    p2 = np.zeros(2**(2*internalnodes + 8))
    pscamarg, pccsc, pacc, pacsc, pccscj, pccs, pcccjs, pcj, pccc, pa, pacccrs, paccps, pacs, ps2,ps, pss, psa2, pct, pscs, psca = calculateDistr(p,internalnodes)
    for i in range(2 ** (2 * internalnodes + 8)):
        p2[i] = pscamarg[i // (2 ** (internalnodes + 4))] * world[
            (i // (2 ** (internalnodes + 2))) % (2 ** 4) + 16 * (i // (2 ** (2 * internalnodes + 6)))]

        for j in range(internalnodes):
            if mode == 2:
                p2[i] = p2[i] * pct[j][(i // (2 ** (internalnodes + 1 - j))) % 2]
            else:
                p2[i] = p2[i] * pccsc[j][ (i // (2 ** (internalnodes + 1 - j))) % 2 + 2 * ((i // (2 ** (internalnodes + 6))))]
        for j in range(2):
            if mode == 1:
                p2[i] = p2[i] * pacc[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (internalnodes + 6)))%(2**internalnodes))]
            if mode == 0:
                p2[i] = p2[i] * pacsc[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (internalnodes + 6))))]
            if mode == 2:
                p2[i] = p2[i] * pacs[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (2*internalnodes + 6))) % (2 ** ( 2)))]

    newppred = np.zeros(16)
    newppred2 = np.zeros(16)
    for i in range(16):
        newppred[i] = ps2[i % 4] * ps[i // 4]
    predic = kl(pss, newppred)
    pit = np.zeros(2 ** (6))
    pit2 = np.zeros(2 ** (6))
    pstat = np.zeros(2 ** 4)
    pstatst = np.zeros(2 ** 6)
    patst2 = np.zeros(2 ** 4)
    pstatst2 = np.zeros(2 ** 6)
    pstst2 = np.zeros(2 ** 4)
    ps3 = np.zeros(4)
    for i in range(2 ** 6):
        pstatst2[i] = psa2[i // 4] * world[i]
    for i in range(2 ** 6):
        ps3[i % 4] = ps3[i % 4] + pstatst2[i]
        patst2[i // 4] = patst2[i // 4] + pstatst2[i]
        pstst2[i % 4 + 4 * (i // 16)] = pstst2[i % 4 + 4 * (i // 16)] + pstatst2[i]
    for i in range(16):
        newppred2[i] = ps2[i // 4] * ps3[i % 4]
    predic2 = kl(pstst2, newppred2)
    pscamarg2, pccsc2, pacc2, pacsc2, pccscj2, pccs2, pcccjs2, pcj2, pccc2, pa2, pacccrs2, paccps2, pacs2, ps2b,psb, pssb, psa2b, pctb, pscs2, psca2= calculateDistr(p2,internalnodes)
    newpworld = np.zeros(2**(8+2*internalnodes))
    newpmorph = np.zeros(2**(8+2*internalnodes))
    newpintinf = np.zeros(2**(8+2*internalnodes))
    newpmemory = np.zeros(2**(8+2*internalnodes))
    newpmemory2 = np.zeros(2**(8+2*internalnodes))
    newpall = np.zeros(2**(8+2*internalnodes))
    newppolicy = np.zeros(2**(8+2*internalnodes))
    newpstrategy = np.zeros(2**(8+2*internalnodes))
    newpmultisenspar = np.zeros(2**(8+2*internalnodes))
    newpmultisenscros = np.zeros(2**(8+2*internalnodes))
    newpworldw= np.zeros(2**(8+2*internalnodes))
    newpworlda=  np.zeros(2**(8+2*internalnodes))

    anewpworld = np.zeros(2**(8+2*internalnodes))
    anewpmorph = np.zeros(2**(8+2*internalnodes))
    anewpintinf = np.zeros(2**(8+2*internalnodes))
    anewpmemory = np.zeros(2**(8+2*internalnodes))
    anewpmemory2 = np.zeros(2**(8+2*internalnodes))
    anewpall = np.zeros(2**(8+2*internalnodes))
    anewppolicy = np.zeros(2**(8+2*internalnodes))
    anewpstrategy = np.zeros(2**(8+2*internalnodes))
    anewpmultisenspar = np.zeros(2**(8+2*internalnodes))
    anewpmultisenscros = np.zeros(2**(8+2*internalnodes))
    anewpworldw= np.zeros(2**(8+2*internalnodes))
    anewpworlda=  np.zeros(2**(8+2*internalnodes))

    #print("psamarg", np.sum(pscamarg), np.min(pscamarg))
    for i in range(2**(8+2*internalnodes)):
        newpworld[i] = psca1[(i // (2**(internalnodes+4)))]  *  ps2[(i//(2**(internalnodes+2)))%(4)]
        newpmorph[i] = psca1[(i // (2**(internalnodes+4)))] * world[(i//(2**(internalnodes +2)))%(2**4) + (2**4)*(i//(2**(2*internalnodes +6)))]
        newpintinf[i] = psca1[(i // (2**(internalnodes+4)))]  * world[(i//(2**(internalnodes +2)))%(2**4) + (2**4)*(i//(2**(2*internalnodes +6)))]
        newpmemory[i] = psca1[(i // (2**(internalnodes+4)))]  * world[(i//(2**(internalnodes +2)))%(2**4) + (2**4)*(i//(2**(2*internalnodes +6)))]
        newpmemory2[i] = psca1[(i // (2**(internalnodes+4)))]  * world[(i//(2**(internalnodes +2)))%(2**4) + (2**4)*(i//(2**(2*internalnodes +6)))]
        newpall[i] =psca1[(i // (2**(internalnodes+4)))] *  ps2[(i//(2**(internalnodes+2)))%(4)]
        newppolicy[i] =psca1[(i // (2**(internalnodes+4)))] * world[(i//(2**(internalnodes +2)))%(2**4) + (2**4)*(i//(2**(2*internalnodes +6)))]
        newpstrategy[i] = psca1[(i // (2**(internalnodes+4)))] * world[(i//(2**(internalnodes +2)))%(2**4) + (2**4)*(i//(2**(2*internalnodes +6)))]
        newpmultisenscros[i] =psca1[(i // (2**(internalnodes+4)))]  * world[(i//(2**(internalnodes +2)))%(2**4) + (2**4)*(i//(2**(2*internalnodes +6)))]
        newpmultisenspar[i] = psca1[(i // (2**(internalnodes+4)))]  * world[(i//(2**(internalnodes +2)))%(2**4) + (2**4)*(i//(2**(2*internalnodes +6)))]
        newpworldw[i] =psca1[(i // (2**(internalnodes+4)))] * pscs[(i // (2 ** (internalnodes + 2))) % 4 + 4 * ((i // (2 ** (2 * internalnodes + 6))) % 4)]
        newpworlda[i] = psca1[(i // (2**(internalnodes+4)))]  * psca[(i // (2 ** (internalnodes + 2))) % 16]

        anewpworld[i] = pscamarg[(i // (2 ** (internalnodes + 4)))] * ps2b[(i // (2 ** (internalnodes + 2))) % (4)]
        anewpmorph[i] = pscamarg[(i // (2 ** (internalnodes + 4)))] * world[(i // (2 ** (internalnodes + 2))) % (2 ** 4) + (2 ** 4) * (i // (2 ** (2 * internalnodes + 6)))]
        anewpintinf[i] = pscamarg[(i // (2 ** (internalnodes + 4)))] * world[(i // (2 ** (internalnodes + 2))) % (2 ** 4) + (2 ** 4) * (i // (2 ** (2 * internalnodes + 6)))]
        anewpmemory[i] = pscamarg[(i // (2 ** (internalnodes + 4)))] * world[(i // (2 ** (internalnodes + 2))) % (2 ** 4) + (2 ** 4) * (i // (2 ** (2 * internalnodes + 6)))]
        anewpmemory2[i] = pscamarg[(i // (2 ** (internalnodes + 4)))] * world[(i // (2 ** (internalnodes + 2))) % (2 ** 4) + (2 ** 4) * (i // (2 ** (2 * internalnodes + 6)))]
        anewpall[i] = pscamarg[(i // (2 ** (internalnodes + 4)))] * ps2b[(i // (2 ** (internalnodes + 2))) % (4)]
        anewppolicy[i] = pscamarg[(i // (2 ** (internalnodes + 4)))] * world[(i // (2 ** (internalnodes + 2))) % (2 ** 4) + (2 ** 4) * (i // (2 ** (2 * internalnodes + 6)))]
        anewpstrategy[i] = pscamarg[(i // (2 ** (internalnodes + 4)))] * world[(i // (2 ** (internalnodes + 2))) % (2 ** 4) + (2 ** 4) * (i // (2 ** (2 * internalnodes + 6)))]
        anewpmultisenscros[i] = pscamarg[(i // (2 ** (internalnodes + 4)))] * world[(i // (2 ** (internalnodes + 2))) % (2 ** 4) + (2 ** 4) * (i // (2 ** (2 * internalnodes + 6)))]
        anewpmultisenspar[i] = pscamarg[(i // (2 ** (internalnodes + 4)))] * world[(i // (2 ** (internalnodes + 2))) % (2 ** 4) + (2 ** 4) * (i // (2 ** (2 * internalnodes + 6)))]
        anewpworldw[i] = pscamarg[(i // (2 ** (internalnodes + 4)))] * pscs2[(i // (2 ** (internalnodes + 2))) % 4 + 4 * ((i // (2 ** (2 * internalnodes + 6))) % 4)]
        anewpworlda[i] = pscamarg[(i // (2 ** (internalnodes + 4)))] * psca2[(i // (2 ** (internalnodes + 2))) % 16]
        for j in range(internalnodes):
            newpworld[i] = newpworld[i] * pccsc[j][(i // (2 ** (internalnodes +1 - j))) % 2 + 2 * (i // (2**(internalnodes +6)))]
            newpmorph[i] = newpmorph[i] * pccsc[j][(i // (2 ** (internalnodes +1 - j))) % 2 + 2 * (i // (2**(internalnodes +6)))]
            newpintinf[i] = newpintinf[i] * pccscj[j][(i // (2 ** ( (internalnodes +1) - j))) % 2 + 2 * ((i // (2**(2*internalnodes +5 -j)))%2) + 4*(i// (2**(6+2*internalnodes)))]
            newpmemory[i] = newpmemory[i] * pccs[j][(i // (2 ** (internalnodes +1 - j))) % 2 + 2 * (i // (2**(2*internalnodes +6)))]
            newpmemory2[i] = newpmemory2[i] * pcccjs[j][(i // (2 ** (internalnodes +1 - j))) % 2 + 2 * ((i // (2 ** (2*internalnodes +6 - j))) % (2 ** (2 + j))) + (2 ** (3 + j)) * ((i // (2 ** (internalnodes +6))) % (2 ** (internalnodes -1 - j)))]
            newpall[i] = newpall[i] * pct[j][(i // (2 ** (internalnodes + 1 - j))) % 2]
            newppolicy[i] = newppolicy[i] * pccsc[j][(i // (2 ** (internalnodes +1 - j))) % 2 + 2 * (i // (2**(internalnodes +6)))]
            newpstrategy[i] = newpstrategy[i] * pccc[j][(i // (2 ** ( (internalnodes + 1) - j))) % 2 + 2 * ((i //  (2**(6+internalnodes)) ) % (2**internalnodes))]
            newpmultisenscros[i] = newpmultisenscros[i]  *pccsc[j][(i // (2 ** (internalnodes +1 - j))) % 2 + 2 * (i // (2**(internalnodes +6)))]
            newpmultisenspar[i] = newpmultisenspar[i] * pccsc[j][(i // (2 ** (internalnodes + 1 - j))) % 2 + 2 * (i // (2 ** (internalnodes + 6)))]
            newpworldw[i] = newpworldw[i] * pccsc[j][(i // (2 ** (internalnodes + 1 - j))) % 2 + 2 * (i // (2 ** (internalnodes + 6)))]
            newpworlda[i] = newpworlda[i] * pccsc[j][(i // (2 ** (internalnodes + 1 - j))) % 2 + 2 * (i // (2 ** (internalnodes + 6)))]

            anewpworld[i] = anewpworld[i] * pccsc2[j][(i // (2 ** (internalnodes + 1 - j))) % 2 + 2 * (i // (2 ** (internalnodes + 6)))]
            anewpmorph[i] = anewpmorph[i] * pccsc2[j][(i // (2 ** (internalnodes + 1 - j))) % 2 + 2 * (i // (2 ** (internalnodes + 6)))]
            anewpintinf[i] = anewpintinf[i] * pccscj2[j][(i // (2 ** ((internalnodes + 1) - j))) % 2 + 2 * ((i // (2 ** (2 * internalnodes + 5 - j))) % 2) + 4 * (i // (2 ** (6 + 2 * internalnodes)))]
            anewpmemory[i] = anewpmemory[i] * pccs2[j][(i // (2 ** (internalnodes + 1 - j))) % 2 + 2 * (i // (2 ** (2 * internalnodes + 6)))]
            anewpmemory2[i] = anewpmemory2[i] * pcccjs2[j][(i // (2 ** (internalnodes + 1 - j))) % 2 + 2 * ((i // (2 ** (2 * internalnodes + 6 - j))) % (2 ** (2 + j))) + (2 ** (3 + j)) * ( (i // (2 ** (internalnodes + 6))) % (2 ** (internalnodes - 1 - j)))]
            anewpall[i] = anewpall[i] * pctb[j][(i // (2 ** (internalnodes + 1 - j))) % 2]
            anewppolicy[i] = anewppolicy[i] * pccsc2[j][(i // (2 ** (internalnodes + 1 - j))) % 2 + 2 * (i // (2 ** (internalnodes + 6)))]
            anewpstrategy[i] = anewpstrategy[i] *pccc2[j][(i // (2 ** ((internalnodes + 1) - j))) % 2 + 2 * ((i // (2 ** (6 + internalnodes))) % (2 ** internalnodes))]
            anewpmultisenscros[i] = anewpmultisenscros[i] * pccsc2[j][(i // (2 ** (internalnodes + 1 - j))) % 2 + 2 * (i // (2 ** (internalnodes + 6)))]
            anewpmultisenspar[i] = anewpmultisenspar[i] * pccsc2[j][(i // (2 ** (internalnodes + 1 - j))) % 2 + 2 * (i // (2 ** (internalnodes + 6)))]
            anewpworldw[i] = anewpworldw[i] * pccsc2[j][(i // (2 ** (internalnodes + 1 - j))) % 2 + 2 * (i // (2 ** (internalnodes + 6)))]
            anewpworlda[i] = anewpworlda[i] * pccsc2[j][(i // (2 ** (internalnodes + 1 - j))) % 2 + 2 * (i // (2 ** (internalnodes + 6)))]
        for j in range(2):
            #controller driven agents
            if mode == 1:
                newpworld[i] = newpworld[i] * pacc[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (internalnodes + 6)))%(2**internalnodes))]
                newpmorph[i] = newpmorph[i] * pacc[j][  (i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (6 + internalnodes))) % (2 ** internalnodes))]
                newpintinf[i] = newpintinf[i] * pacc[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (internalnodes + 6)))%(2**internalnodes))]
                newpmemory[i] = newpmemory[i] * pacc[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (internalnodes + 6)))%(2**internalnodes))]
                newpmemory2[i] = newpmemory2[i] * pacc[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (internalnodes + 6)))%(2**internalnodes))]
                newpall[i] = newpall[i] * pa[j][(i // (2 ** (1 - j))) % 2]
                newppolicy[i] = newppolicy[i] * pa[j][(i // (2 ** ( 1 - j))) % 2]
                newpstrategy[i] = newpstrategy[i] * pacc[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (internalnodes + 6)))%(2**internalnodes))]
                newpmultisenscros[i] = newpmultisenscros[i] * pacccrs[j][ (i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (6 + internalnodes))) % (2 ** internalnodes)) + 2 ** (internalnodes + 1) * ((i // (2 ** (2 * internalnodes + 6 + j))) % 2)]
                newpmultisenspar[i] = newpmultisenspar[i] * paccps[j][ (i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (6 + internalnodes))) % (2 ** internalnodes)) + 2 ** (internalnodes + 1) * ((i // (2 ** (2 * internalnodes + 7 - j))) % 2)]
                newpworldw[i] = newpworldw[i] * pacc[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (internalnodes + 6)))%(2**internalnodes))]
                newpworlda[i] = newpworlda[i] * pacc[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (internalnodes + 6)))%(2**internalnodes))]

                anewpworld[i] = anewpworld[i] * pacc2[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (internalnodes + 6)))%(2**internalnodes))]
                anewpmorph[i] = anewpmorph[i] * pacc2[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (internalnodes + 6)))%(2**internalnodes))]
                anewpintinf[i] = anewpintinf[i] * pacc2[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (internalnodes + 6)))%(2**internalnodes))]
                anewpmemory[i] = anewpmemory[i] * pacc2[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (internalnodes + 6)))%(2**internalnodes))]
                anewpmemory2[i] = anewpmemory2[i] * pacc2[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (internalnodes + 6)))%(2**internalnodes))]
                anewpall[i] = anewpall[i] * pa2[j][(i // (2 ** (1 - j))) % 2]
                anewppolicy[i] = anewppolicy[i] * pa2[j][(i // (2 ** (1 - j))) % 2]
                anewpstrategy[i] = anewpstrategy[i] * pacc2[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (internalnodes + 6)))%(2**internalnodes))]
                anewpmultisenscros[i] = anewpmultisenscros[i] * pacccrs2[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (6 + internalnodes))) % (2 ** internalnodes)) + 2 ** (internalnodes + 1) * ((i // (2 ** (2 * internalnodes + 6 + j))) % 2)]
                anewpmultisenspar[i] = anewpmultisenspar[i] * paccps2[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (6 + internalnodes))) % (2 ** internalnodes)) + 2 ** (internalnodes + 1) * ((i // (2 ** (2 * internalnodes + 7 - j))) % 2)]
                anewpworldw[i] = anewpworldw[i] * pacc2[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (internalnodes + 6)))%(2**internalnodes))]
                anewpworlda[i] = anewpworlda[i] * pacc2[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (internalnodes + 6)))%(2**internalnodes))]
            #reactive control agents
            if mode == 2:
                newpworld[i] = newpworld[i] * pacs[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (2*internalnodes + 6))) % (2 ** ( 2)))]
                newpmorph[i] = newpmorph[i] * pa[j][(i // (2 ** (1 - j))) % 2]
                newpintinf[i] = newpintinf[i] * pacs[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (2*internalnodes + 6))) % (2 ** ( 2)))]
                newpmemory[i] = newpmemory[i] * pacs[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (2*internalnodes + 6))) % (2 ** ( 2)))]
                newpmemory2[i] = newpmemory2[i] * pacs[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (2*internalnodes + 6))) % (2 ** ( 2)))]
                newpall[i] = newpall[i] * pa[j][(i // (2 ** (1 - j))) % 2]
                newppolicy[i] = newppolicy[i] * pacs[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (2*internalnodes + 6))) % (2 ** ( 2)))]
                newpstrategy[i] = newpstrategy[i] * pacs[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (2*internalnodes + 6))) % (2 ** ( 2)))]
                newpmultisenscros[i] = newpmultisenscros[i] * pacccrs[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (6 + internalnodes))) % (2 ** internalnodes)) + 2 ** (internalnodes + 1) * ((i // (2 ** (2 * internalnodes + 6 + j))) % 2)]
                newpmultisenspar[i] = newpmultisenspar[i] * paccps[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (6 + internalnodes))) % (2 ** internalnodes)) + 2 ** (internalnodes + 1) * ((i // (2 ** (2 * internalnodes + 7 - j))) % 2)]
                newpworldw[i] = newpworldw[i] * pacs[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (2*internalnodes + 6))) % (2 ** ( 2)))]
                newpworlda[i] = newpworlda[i] * pacs[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (2*internalnodes + 6))) % (2 ** ( 2)))]

                anewpworld[i] = anewpworld[i] * pacs2[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (2*internalnodes + 6))) % (2 ** ( 2)))]
                anewpmorph[i] = anewpmorph[i] * pa2[j][(i // (2 ** (1 - j))) % 2]
                anewpintinf[i] = anewpintinf[i] * pacs2[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (2*internalnodes + 6))) % (2 ** ( 2)))]
                anewpmemory[i] = anewpmemory[i] * pacs2[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (2*internalnodes + 6))) % (2 ** ( 2)))]
                anewpmemory2[i] = anewpmemory2[i] * pacs2[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (2*internalnodes + 6))) % (2 ** ( 2)))]
                anewpall[i] = anewpall[i] * pa2[j][(i // (2 ** (1 - j))) % 2]
                anewppolicy[i] = anewppolicy[i] * pacs2[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (2*internalnodes + 6))) % (2 ** ( 2)))]
                anewpstrategy[i] = anewpstrategy[i] * pacs2[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (2*internalnodes + 6))) % (2 ** ( 2)))]
                anewpmultisenscros[i] = anewpmultisenscros[i] * pacccrs2[j][ (i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (6 + internalnodes))) % (2 ** internalnodes)) + 2 ** (internalnodes + 1) * ((i // (2 ** (2 * internalnodes + 6 + j))) % 2)]
                anewpmultisenspar[i] = anewpmultisenspar[i] * paccps2[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (6 + internalnodes))) % (2 ** internalnodes)) + 2 ** (internalnodes + 1) * ((i // (2 ** (2 * internalnodes + 7 - j))) % 2)]
                anewpworldw[i] = anewpworldw[i] * pacs2[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (2*internalnodes + 6))) % (2 ** ( 2)))]
                anewpworlda[i] = anewpworlda[i] * pacs2[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (2*internalnodes + 6))) % (2 ** ( 2)))]
            #full agents
            if mode == 0:
                newpworld[i] = newpworld[i] * pacsc[j][(i // (2 ** (1 - j))) % 2 + 2 * (i // (2**(6+internalnodes)))]
                newpmorph[i] = newpmorph[i] * pacc[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i //(2**(6+internalnodes))) % (2**internalnodes))]
                newpintinf[i] = newpintinf[i] * pacsc[j][(i // (2 ** (1 - j))) % 2 + 2 * (i // (2**(6+internalnodes)))]
                newpmemory[i] = newpmemory[i] * pacsc[j][(i // (2 ** (1 - j))) % 2 + 2 * (i // (2**(6+internalnodes)))]
                newpmemory2[i] = newpmemory2[i] * pacsc[j][(i // (2 ** (1 - j))) % 2 + 2 * (i // (2**(6+internalnodes)))]
                newpall[i] = newpall[i] * pa[j][(i // (2 ** (1 - j))) % 2]
                newppolicy[i] = newppolicy[i] * pacs[j][(i // (2 ** (1 - j))) % 2 + 2 * (i //  (2**(6+2*internalnodes)))]
                newpstrategy[i] = newpstrategy[i] * pacsc[j][(i // (2 ** (1 - j))) % 2 + 2 * (i // (2**(6+internalnodes)))]
                newpmultisenscros[i] = newpmultisenscros[i] *pacccrs[j][(i//(2**(1-j)))%2 + 2 * ((i //(2**(6+internalnodes))) % (2**internalnodes)) + 2**(internalnodes +1)*((i//(2**(2*internalnodes + 6 +j)))%2 )]
                newpmultisenspar[i] = newpmultisenspar[i] * paccps[j][(i//(2**(1-j)))%2 + 2 * ((i //(2**(6+internalnodes))) % (2**internalnodes)) + 2**(internalnodes +1)*((i//(2**(2*internalnodes + 7 -j)))%2 )]
                newpworldw[i] = newpworldw[i] * pacsc[j][(i // (2 ** (1 - j))) % 2 + 2 * (i // (2 ** (6 + internalnodes)))]
                newpworlda[i] = newpworlda[i] * pacsc[j][(i // (2 ** (1 - j))) % 2 + 2 * (i // (2 ** (6 + internalnodes)))]

                anewpworld[i] = anewpworld[i] * pacsc2[j][(i // (2 ** (1 - j))) % 2 + 2 * (i // (2**(6+internalnodes)))]
                anewpmorph[i] = anewpmorph[i] * pacc2[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i //(2**(6+internalnodes))) % (2**internalnodes))]
                anewpintinf[i] = anewpintinf[i] * pacsc2[j][(i // (2 ** (1 - j))) % 2 + 2 * (i // (2**(6+internalnodes)))]
                anewpmemory[i] = anewpmemory[i] * pacsc2[j][(i // (2 ** (1 - j))) % 2 + 2 * (i // (2**(6+internalnodes)))]
                anewpmemory2[i] = anewpmemory2[i] * pacsc2[j][(i // (2 ** (1 - j))) % 2 + 2 * (i // (2**(6+internalnodes)))]
                anewpall[i] = anewpall[i] * pa2[j][(i // (2 ** (1 - j))) % 2]
                anewppolicy[i] = anewppolicy[i] * pacs2[j][(i // (2 ** (1 - j))) % 2 + 2 * (i //(2**(6+2*internalnodes)))]
                anewpstrategy[i] = anewpstrategy[i] * pacsc2[j][(i // (2 ** (1 - j))) % 2 + 2 * (i // (2**(6+internalnodes)))]
                anewpmultisenscros[i] = anewpmultisenscros[i] *pacccrs2[j][(i//(2**(1-j)))%2 + 2 * ((i //(2**(6+internalnodes))) % (2**internalnodes)) + 2**(internalnodes +1)*((i//(2**(2*internalnodes + 6 +j)))%2 )]
                anewpmultisenspar[i] = anewpmultisenspar[i] * paccps2[j][(i//(2**(1-j)))%2 + 2 * ((i //(2**(6+internalnodes))) % (2**internalnodes)) + 2**(internalnodes +1)*((i//(2**(2*internalnodes + 7 -j)))%2 )]
                anewpworldw[i] = anewpworldw[i] * pacsc2[j][(i // (2 ** (1 - j))) % 2 + 2 * (i // (2 ** (6 + internalnodes)))]
                anewpworlda[i] = anewpworlda[i] * pacsc2[j][(i // (2 ** (1 - j))) % 2 + 2 * (i // (2 ** (6 + internalnodes)))]
    worldval = kl(p0, newpworld)
    worldval2 = kl(p2, anewpworld)
    multi = min(kl(p0, newpmultisenscros),  kl(p0, newpmultisenspar))
    multi2 = min(kl(p2, anewpmultisenscros), kl(p2, anewpmultisenspar))
    morph = kl(p0, newpmorph)
    morph2 = kl(p2, anewpmorph)
    intinf = kl(p0, newpintinf)
    intinf2 = kl(p2, anewpintinf)
    memory = kl(p0, newpmemory)
    memory21 = kl(p2, anewpmemory)
    memory2 = kl(p0, newpmemory2)
    memory22 = kl(p2, anewpmemory2)
    all = kl(p0, newpall)
    all2 = kl(p2, anewpall)
    policy = kl(p0, newppolicy)
    policy2= kl(p2, anewppolicy)
    strategy = kl(p0, newpstrategy)
    strategy2 = kl(p2, anewpstrategy)
    worldwa = kl(p0, newpworlda)
    worldwa2 = kl(p2, anewpworlda)
    worldww = kl(p0, newpworldw)
    worldww2 = kl(p2, anewpworldw)
    worldww = 1- (1/log2(4))*worldww
    worldww2= 1-(1/log2(4))*worldww2
    for i in range(2**(8+2*internalnodes)):
        pstat[(i//(2**(internalnodes+4)) )%4 + 4*((i//(2**(2*internalnodes+6)))%4) ] = pstat[(i//(2**(internalnodes+4)) )%4 + 4*((i//(2**(2*internalnodes+6)))%4) ] +p[i]
        pstatst[(i//(2**(internalnodes+2)) )%16 + 16*((i//(2**(2*internalnodes+6)))%4) ] = pstatst[(i//(2**(internalnodes+2)) )%16 + 16*((i//(2**(2*internalnodes+6)))%4) ] + p[i]
    for i in range(2**6):
        pit[i] = pstat[i//4] * 0.25
        pit2[i] = patst2[i//4] * 0.25
  ##iterative scaling algorithm for Psi_syn
    for z in range(100):
        if z%2==0:
            pitscs= np.zeros(2 ** 4)
            pitsw = np.zeros(4)
            pitscs2 = np.zeros(2 ** 4)
            pitsw2 = np.zeros(4)
            for i in range(2**(6)):
                pitscs[(i %4) + 4*(i//(2**4))] = pitscs[(i %4) + 4*(i//(2**4))] + pit[i]
                pitsw[i// (2**4)] = pitsw[i// (2**4)] + pit[i]
                pitscs2[(i % 4) + 4 * (i // (2 ** 4))] = pitscs2[(i % 4) + 4 * (i // (2 ** 4))] + pit2[i]
                pitsw2[i // (2 ** 4)] = pitsw2[i // (2 ** 4)] + pit2[i]
            for i in range(16):
                if pitsw[i // 4] != 0:
                    pitscs[i] = pitscs[i] / pitsw[i // 4]
                else:
                    pitscs[i] = 1 / 4
                if pitsw2[i // 4] != 0:
                    pitscs2[i] = pitscs2[i] / pitsw2[i // 4]
                else:
                    pitscs2[i] = 1 / 4
            for i in range(2**(6)):
                pit[i] = pit[i] * (pscs[(i %4) + 4*(i//(2**4))]/ pitscs[(i %4) + 4*(i//(2**4))])
                pit2[i] = pit2[i] * (pscs[(i %4) + 4*(i//(2**4))]/ pitscs2[(i %4) + 4*(i//(2**4))])
        else:
            pitsca = np.zeros(2**4)
            pitaw= np.zeros(4)
            for i in range(2 ** (6)):
                pitsca[i %16 ] = pitsca[i%16] + pit[i]
                pitaw[(i//(2**(2)))%4] = pitaw[(i//(2**(2)))%4] +pit[i]
            for i in range(16):
                if pitaw[i // 4] != 0:
                    pitsca[i] = pitsca[i] / pitaw[i // 4]
                else:
                    pitsca[i] = 1 / 4
            for i in range(2**(6)):
                pit[i] = pit[i] *(psca[i  % 16] / pitsca[i  % 16])
            pitsca2 = np.zeros(2 ** 4)
            pitaw2 = np.zeros(4)
            for i in range(2 ** (6)):
                pitsca2[i % 16] = pitsca2[i % 16] + pit2[i]
                pitaw2[(i // (2 ** (2))) % 4] = pitaw2[(i // (2 ** (2))) % 4] + pit2[i]
            for i in range(16):
                if pitaw2[i // 4] != 0:
                    pitsca2[i] = pitsca2[i] / pitaw2[i // 4]
                else:
                    pitsca2[i] = 1 / 4
            for i in range(2 ** (6)):
                pit2[i] = pit2[i] * (psca[i % 16] / pitsca2[i % 16])

    return worldval, morph, intinf, memory, memory2, all, policy, strategy, multi, worldwa, worldww, kl(pstatst,pit), predic, worldval2, morph2, intinf2, memory21, memory22, all2, policy2, strategy2, multi2, worldwa2, worldww2, kl(pstatst2,pit2), predic2
        # \psi_SA, \psi_R, \phi_T, \psi_M, \psi_M2, \psi_TIF, \psi_C , \psi_SI, \psi_MSI, \psi_S , \psi_A, \psi_syn, \psi_PI
        #      0      1       2       3       4       5        6          7        8          9       10       11       12

#calculating marginal distributions for the calculation of the measures
def calculateDistr(p, internalnodes):
    pscamarg = np.zeros(2 ** (internalnodes + 4))
    psc = np.zeros(2 ** (2 + internalnodes))
    pss = np.zeros(2 ** 4)
    ps2 = np.zeros(2 ** 2)
    psa2 = np.zeros(2 ** 4)
    pa2 = np.zeros(4)
    pa = np.zeros((2, 2))
    pacsc = np.zeros((2, 2 ** (3 + internalnodes)))
    pacc = np.zeros((2, 2 ** (1 + internalnodes)))
    pacs = np.zeros((2, 8))
    paccps = np.zeros((2, 2 ** (internalnodes + 2)))
    pacccrs = np.zeros((2, 2 ** (internalnodes + 2)))
    pccsc = np.zeros((internalnodes, 2 ** (3 + internalnodes)))
    pccc = np.zeros((internalnodes, 2 ** (1 + internalnodes)))
    pccscj = np.zeros((internalnodes, 16))
    pcccj = np.zeros((internalnodes, 4))
    pccs = np.zeros((internalnodes, 8))
    pcccjs = np.zeros((internalnodes, 2 ** (2 + internalnodes)))
    pcjs = np.zeros((internalnodes, 2 ** (1 + internalnodes)))
    pcps = np.zeros((2, 2 ** (internalnodes + 1)))
    ps = np.zeros(4)
    pcj = np.zeros((internalnodes, 2))
    pc = np.zeros(2 ** internalnodes)
    pct = np.zeros((internalnodes, 2))
    pscj = np.zeros((internalnodes, 8))
    pscs = np.zeros(2**4)
    psca = np.zeros(2**4)
    for i in range(2 ** (8 + 2 * internalnodes)):
        ps2[(i // (2 ** (internalnodes + 2))) % 4] = ps2[(i // (2 ** (internalnodes + 2))) % 4] + p[i]
        pscs[(i//(2**(internalnodes+2)))%4 + 4* (i//(2**(2*internalnodes + 6)))] = pscs[(i//(2**(internalnodes+2)))%4 + 4* (i//(2**(2*internalnodes + 6)))] + p[i]
        psca[(i // (2 ** (internalnodes + 2))) % 16 ] = psca[(i // ( 2 ** (internalnodes + 2))) % 16] + p[i]
        pscamarg[i % (2 ** (internalnodes + 4))] = pscamarg[i % (2 ** (internalnodes + 4))] + p[i]
        psc[i // (2 ** (6 + internalnodes))] = psc[i // (2 ** (6 + internalnodes))] + p[i]
        pc[(i // (2 ** (6 + internalnodes))) % (2 ** internalnodes)] = pc[(i // (2 ** (6 + internalnodes))) % (2 ** internalnodes)] + p[i]
        ps[(i // (2 ** (6 + 2 * internalnodes)))] = ps[(i // (2 ** (6 + 2 * internalnodes)))] + p[i]
        psa2[i % 4 + 4 * (i // (2 ** (internalnodes + 2)) % 4)] = psa2[i % 4 + 4 * ( i // (2 ** (internalnodes + 2)) % 4)] + p[i]
        pss[(i // (2 ** (internalnodes + 2))) % 4 + 4 * ((i // (2 ** (6 + 2 * internalnodes))))] = pss[(i // ( 2 ** (internalnodes + 2))) % 4 + 4 * ((i // (2 ** (6 + 2 * internalnodes))))] + p[i]
        pa2[(i//(2**(internalnodes +4)))%4] = pa2[(i//(2**(internalnodes +4)))%4] + p[i]
        for j in range(2):
            pacsc[j][(i // (2 ** (1 - j))) % 2 + 2 * (i // (2 ** (6 + internalnodes)))] = pacsc[j][(i // (2 ** (1 - j))) % 2 + 2 * (i // (2 ** (6 + internalnodes)))] + p[i]
            pacc[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (6 + internalnodes))) % (2 ** internalnodes))] = pacc[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (6 + internalnodes))) % (2 ** internalnodes))] + p[i]
            pacs[j][(i // (2 ** (1 - j))) % 2 + 2 * (i // (2 ** (6 + 2 * internalnodes)))] = pacs[j][(i // ( 2 ** (1 - j))) % 2 + 2 * (i // (2 ** (6 + 2 * internalnodes)))] + p[i]
            pa[j][(i // (2 ** (1 - j))) % 2] = pa[j][(i // (2 ** (1 - j))) % 2] + p[i]
            paccps[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (6 + internalnodes))) % (2 ** internalnodes)) + 2 ** (internalnodes + 1) * ((i // (2 ** (2 * internalnodes + 7 - j))) % 2)] = paccps[j][(i // (
                        2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (6 + internalnodes))) % (2 ** internalnodes)) + 2 ** (internalnodes + 1) * (( i // (2 ** (2 * internalnodes + 7 - j))) % 2)] +  p[i]
            pacccrs[j][(i // (2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (6 + internalnodes))) % (2 ** internalnodes)) + 2 ** (internalnodes + 1) * ((i // (2 ** (2 * internalnodes + 6 + j))) % 2)] = pacccrs[j][(i // (
                        2 ** (1 - j))) % 2 + 2 * ((i // (2 ** (6 + internalnodes))) % (2 ** internalnodes)) + 2 ** (internalnodes + 1) * ( (i // ( 2 ** (2 * internalnodes + 6 + j))) % 2)] +  p[i]
            pcps[j][(i // (2 ** (6 + internalnodes))) % (2 ** internalnodes) + (2 ** internalnodes) * ((i // (2 ** (2 * internalnodes + 7 - j))) % 2)] = pcps[j][(i // (2 ** (6 + internalnodes))) % ( 2 ** internalnodes) + (2 ** internalnodes) * ((i // (2 ** (2 * internalnodes + 7 - j))) % 2)] +  p[i]
        for j in range(internalnodes):
            pct[j][(i // (2 ** (internalnodes + 1 - j))) % 2] = pct[j][(i // (2 ** (internalnodes + 1 - j))) % 2] + p[i]
            pccsc[j][(i // (2 ** ((internalnodes + 1) - j))) % 2 + 2 * (i // (2 ** (6 + internalnodes)))] = pccsc[j][(i // (2 ** ( (internalnodes + 1) - j))) % 2 + 2 * (i // (2 ** (6 + internalnodes)))] +  p[i]
            pccc[j][(i // (2 ** ((internalnodes + 1) - j))) % 2 + 2 * ((i // (2 ** (6 + internalnodes))) % (2 ** internalnodes))] = pccc[j][(i // (2 ** ((internalnodes + 1) - j))) % 2 + 2 * ((i // (2 ** (6 + internalnodes))) % ( 2 ** internalnodes))] + p[i]
            pccscj[j][(i // (2 ** ((internalnodes + 1) - j))) % 2 + 2 * ((i // (2 ** (2 * internalnodes + 5 - j))) % 2) + 4 * (i // (2 ** (6 + 2 * internalnodes)))] = pccscj[j][(i // (2 ** ((internalnodes + 1) - j))) % 2 + 2 * ((i // (2 ** (2 * internalnodes + 5 - j))) % 2) + 4 * (i // (2 ** (6 + 2 * internalnodes)))] + p[ i]
            pscj[j][(i // (2 ** (2 * internalnodes + 5 - j))) % 2 + 2 * (i // (2 ** (6 + 2 * internalnodes)))] = pscj[j][(i // (2 ** (2 * internalnodes + 5 - j))) % 2 + 2 * (i // (2 ** (6 + 2 * internalnodes)))] + p[i]
            pcccj[j][(i // (2 ** (internalnodes + 1 - j))) % 2 + 2 * ((i // (2 ** (2 * internalnodes + 5 - j))) % 2)] = pcccj[j][(i // (2 ** (internalnodes + 1 - j))) % 2 + 2 * ((i // (2 ** (2 * internalnodes + 5 - j))) % 2)] + p[i]
            pcj[j][(i // (2 ** (2 * internalnodes + 5 - j))) % 2] = pcj[j][(i // (2 ** (2 * internalnodes + 5 - j))) % 2] +  p[i]
            pccs[j][(i // (2 ** (internalnodes + 1 - j))) % 2 + 2 * (i // (2 ** (2 * internalnodes + 6)))] = pccs[j][(i // ( 2 ** (internalnodes + 1 - j))) % 2 + 2 * (i // (2 ** (  2 * internalnodes + 6)))] + p[i]
            pcjs[j][((i // (2 ** (2 * internalnodes + 6 - j))) % (2 ** (2 + j))) + (2 ** (2 + j)) * ( (i // (2 ** (internalnodes + 6))) % (2 ** (internalnodes - 1 - j)))] = pcjs[j][((i // (2 ** (2 * internalnodes + 6 - j))) % (2 ** (2 + j))) + (2 ** (2 + j)) * ((i // (2 ** (internalnodes + 6))) % (2 ** (internalnodes - 1 - j)))] + p[i]
            pcccjs[j][(i // (2 ** (internalnodes + 1 - j))) % 2 + 2 * ((i // (2 ** (2 * internalnodes + 6 - j))) % (2 ** (2 + j))) + (2 ** (3 + j)) * ( (i // (2 ** (internalnodes + 6))) % (2 ** (internalnodes - 1 - j)))] = pcccjs[j][( i // ( 2 ** (internalnodes + 1 - j))) % 2 + 2 * (( i // ( 2 ** ( 2 * internalnodes + 6 - j))) % (2 ** (  2 + j))) + (  2 ** (  3 + j)) * ( ( i // ( 2 ** ( internalnodes + 6))) % (2 ** (internalnodes - 1 - j)))] +  p[i]

    for i in range(2 ** (3 + internalnodes)):
        if psc[i // 2] != 0:
            for j in range(2):
                pacsc[j][i] = pacsc[j][i] / psc[i // 2]
            for j in range(internalnodes):
                pccsc[j][i] = pccsc[j][i] / psc[i // 2]
        else:
            for j in range(2):
                pacsc[j][i] = 0.5
            for j in range(internalnodes):
                pccsc[j][i] = 0.5
 #   print(pacsc, np.sum(pacsc))

    for i in range(2 ** (internalnodes + 2)):
        for j in range(2):
            if pcps[j][i // 2] != 0:
                paccps[j][i] = paccps[j][i] / pcps[j][i // 2]
            else:
                paccps[j][i] = 0.5
            if pcps[1 - j][i // 2] != 0:
                pacccrs[j][i] = pacccrs[j][i] / pcps[1 - j][i // 2]
            else:
                pacccrs[j][i] = 0.5

    for i in range(2 ** (2 + internalnodes)):
        for j in range(internalnodes):
            if pcjs[j][i // 2] != 0:
                pcccjs[j][i] = pcccjs[j][i] / pcjs[j][i // 2]
            else:
                pcccjs[j][i] = 0.5

    for i in range(2 ** (1 + internalnodes)):
        for j in range(2):
            if pc[i // 2] != 0:
                pacc[j][i] = pacc[j][i] / pc[i // 2]
            else:
                pacc[j][i] = 0.5
        for j in range(internalnodes):
            if pc[i // 2] != 0:
                pccc[j][i] = pccc[j][i] / pc[i // 2]
            else:
                pccc[j][i] = 0.5

    for i in range(16):
        for j in range(internalnodes):
            if pscj[j][i // 2] != 0:
                pccscj[j][i] = pccscj[j][i] / pscj[j][i // 2]
            else:
                pccscj[j][i] = 0.5
        if ps[i//4] !=0:
            pscs[i] = pscs[i] / ps[i //4]
        else:
            pscs[i] = 0.25
        if pa2[i//4] !=0:
            psca[i] = psca[i] / pa2[i//4]
        else:
            psca[i] = 0.25

    for i in range(8):
        if ps[i // 2] != 0:
            for j in range(2):
                if pacs[j][i] != 0:
                    pacs[j][i] = pacs[j][i] / ps[i // 2]
                else:
                    print("here", pacs, np.sum(pacs))
            for j in range(internalnodes):
                pccs[j][i] = pccs[j][i] / ps[i // 2]
        else:
            for j in range(2):
                pacs[j][i] = 0.5
            for j in range(internalnodes):
                pccs[j][i] = 0.5
    for i in range(4):
        for j in range(internalnodes):
            if pcj[j][i // 2] != 0:
                pcccj[j][i] = pcccj[j][i] / pcj[j][i // 2]
            else:
                pcccj[j][i] = 0.5
    return pscamarg, pccsc, pacc, pacsc, pccscj, pccs, pcccjs, pcj, pccc, pa, pacccrs, paccps, pacs, ps2,ps, pss, psa2, pct, pscs, psca


# kl divergence
def kl(y,x):
    sumkl = 0.0
    for i in range(len(y)):
        if y[i] > 0:
            if x[i] > 0:
                sumkl = sumkl + y[i] * (log2(y[i]) - log2(x[i]))
            else:
                print(x[i], y[i])
                #sumkl = sumkl + 100
                x[i] = 1*10**-323
                sumkl = sumkl + y[i] * (log2(y[i]) - log2(x[i]))
    return sumkl

#get random initial distribution
def randdistr(internalnodes):
    p = np.zeros(2**(9+2*internalnodes))
    for i in range(2**(9+2*internalnodes)):
        p[i] = random.randint(5,10000)
    p = p/np.sum(p)
    return p