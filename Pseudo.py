###############################################################################
# Pseudo Dataset
###############################################################################

import numpy as np
import random
import copy
import networkx as nx
import dimod
from dimod.reference.samplers import ExactSolver

class PseudoDataset:
    
    def __init__(self, N_sub=5, Lambda_max=5, N_G=20, score0=100, iter_max=100):
        self.N_sub = N_sub
        self.Lambda_max = Lambda_max
        self.N_G = N_G
        self.score0 = score0
        self.iter_max = iter_max
    
    def parent(self, Lambda, N0, N0_, n):
        p1n = []
        while len(p1n) < Lambda - 1:
            Prob_sub = random.random()
            hoge = set([m for m in range(N0, N0_) if random.random() >= Prob_sub and m != n])
            if hoge != set():
                for p1n_ in copy.copy(p1n):
                    if hoge <= p1n_ or hoge >= p1n_:
                        p1n.remove(p1n_)
                p1n += [hoge]
        return p1n

    def subgraph(self, N0):
        p1_sub = []
        s0_sub = [self.score0 for n in range(self.N_sub)]
        s1_sub = []
        for n in range(N0, N0 + self.N_sub):    
            Lambda = random.choice(range(1, self.Lambda_max + 1))
            p1n = self.parent(Lambda, N0, N0 + self.N_sub, n)
            p1_sub += [p1n]
            s1_sub += [[-self.score0 * random.random() for lambda_ in range(Lambda - 1)]]
        s0_sum = sum(s0_sub)
        Q = dict()
        for n in range(N0, N0 + self.N_sub):
            for i in range(len(p1_sub[n - N0])):
                Q.update({((i,n),(i,n)): s1_sub[n - N0][i]})
            for i in range(len(p1_sub[n - N0])-1):
                for ii in range(i+1,len(p1_sub[n - N0])):
                    Q.update({((i,n), (ii,n)): 10000000000})
        bqm = dimod.BinaryQuadraticModel.from_qubo(Q, s0_sum)
        response = ExactSolver().sample(bqm)
        G_sub = []
        S_sub = []
        for res in response:
            edges = []
            const = 0
            s = s0_sum
            for n in range(N0, N0 + self.N_sub):
                tmp = 0
                for i in range(len(p1_sub[n - N0])):
                    if res[(i,n)] == 1:
                        for m in list(p1_sub[n - N0][i]):
                            edges += [(m,n)]
                        s += s1_sub[n - N0][i]
                        tmp += 1
                if tmp > 1:
                    const += 1
            G = nx.DiGraph(edges)
            if const == 0 and nx.is_directed_acyclic_graph(G) == True:
                G_sub += [edges]
                S_sub += [s]
        return G_sub, S_sub, p1_sub, s1_sub
    
    def initial(self):       
        G = []
        S = []
        p1 = []
        s1 = []
        for ng in range(self.N_G):
            G_sub, S_sub, p1_sub, s1_sub = self.subgraph(self.N_sub * ng)
            G += [G_sub]
            S += [S_sub]
            p1 += p1_sub
            s1 += s1_sub
        return G, S, p1, s1
    
    def generate(self):
        N = self.N_sub * self.N_G
        p2 = [[] for n in range(N)]
        s0 = [self.score0 for n in range(N)]
        s2 = [[] for n in range(N)]
        t = [[[]] for n in range(N)]
        G, S, p1, s1 = self.initial()
        G0 = []
        score = 0        
        for ng in range(self.N_G):
            if G[ng] != []:                
                G0 += G[ng][0]
                score += S[ng][0]
            else:
                score += self.score0 * self.N_sub
        for n in range(len(p1)):
            iter_ = 0
            while p1[n] == []:
                Lambda = random.choice(range(1, self.Lambda_max + 1))
                p1n = self.parent(Lambda, 0, N, n)
                s1n = [-self.score0 * random.random() for lambda_ in range(Lambda - 1)]
                if s1n != []:
                    G0_ = copy.copy(G0)
                    for m in list(p1n[np.argmin(s1n)]):
                        G0_ += [(m, n)]
                    if nx.is_directed_acyclic_graph(nx.DiGraph(G0_)) == True:
                        G0 = copy.copy(G0_)
                        score += min(s1n)
                        p1[n] += p1n
                        s1[n] += s1n
                iter_ += 1
                if iter_ == self.iter_max:
                    break
        delta_lower = 0
        for s1_ in s1:
            if s1_ != []:
                delta_lower = max(delta_lower, -min(s1_))
        return p1, p2, s0, s1, s2, t, delta_lower, score, G0