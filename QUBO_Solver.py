###############################################################################
# QUBO Solver
###############################################################################

import numpy as np
import networkx as nx
import pylab
import neal
import dimod
from dimod.reference.samplers import ExactSolver
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from dwave_qbsolv import QBSolv
#from dwave.cloud import Client

class QUBOSolver:
    
    def __init__(self, stress=1000):
        self.stress = stress

    def Hamiltonian(self, p1, p2, s0, s1, s2, t, delta_lower):
        self.p1 = p1
        self.p2 = p2
        self.s0 = s0
        self.s1 = s1
        self.s2 = s2        
        self.t = t 
        self.N = len(p1)
        delta1 = delta_lower + self.stress 
        delta2 = delta_lower * 3 + self.stress 
        delta3 = delta2 * (self.N - 2) / 3 + self.stress 
        self.s0_sum = sum(s0)
        self.Q = dict()
        for n in range(self.N):
            if p1[n] != []:
                for i in range(len(p1[n])):
                    if n > 0:
                        c = sum([1 for n_ in range(n) if n_ in p1[n][i]])
                    else:
                        c = 0
                    self.Q.update({((1,i,n),(1,i,n)): s1[n][i] + delta3 * c})
            if p2[n] != []:
                for i in range(len(p2[n])):
                    if n > 0:
                        c = sum([1 for n_ in range(n) if n_ in p2[n][i]])
                    else:
                        c = 0
                    self.Q.update({((2,i,n),(2,i,n)): s2[n][i] + delta3 * c})
            if p1[n] != [] and p2[n] != []:
                for i in range(len(p1[n])):
                    for j in range(len(p2[n])):
                        self.Q.update({((1,i,n),(2,j,n)): t[n][i][j]})
            if len(p1[n]) >= 2:
                for i in range(len(p1[n])-1):
                    for ii in range(i+1,len(p1[n])):
                        self.Q.update({((1,i,n), (1,ii,n)): delta1})
            if len(p2[n]) >= 2:
                for j in range(len(p2[n])-1):
                    for jj in range(j+1,len(p2[n])):
                        self.Q.update({((2,j,n), (2,jj,n)): delta1})
        tmp = np.zeros((self.N, self.N))
        for n in range(self.N-2):
            for nn in range(n+1,self.N-1):
                for nnn in range(nn+1,self.N):
                    tmp[n, nnn] += delta2
                    self.Q.update({((n,nnn), (nn,nnn)): -delta2})
                    self.Q.update({((n,nnn), (n,nn)): -delta2})
                    self.Q.update({((n,nn), (nn,nnn)): delta2})
        for n in range(self.N-2):
            for nnn in range(n+2,self.N):
                self.Q.update({((n,nnn), (n,nnn)): tmp[n, nnn]})        
        for n in range(1,self.N):
            for n_ in range(n):
                if p1[n_] != []:
                    for i in range(len(p1[n_])):
                        if n in p1[n_][i]:
                            self.Q.update({((1,i,n_),(n_,n)): delta3})
                if p1[n] != []:
                    for i in range(len(p1[n])):
                        if n_ in p1[n][i]:
                            self.Q.update({((1,i,n),(n_,n)): - delta3})
                if p2[n_] != []:
                    for i in range(len(p2[n_])):
                        if n in p2[n_][i]:
                            self.Q.update({((2,i,n_),(n_,n)): delta3})
                if p2[n] != []:
                    for i in range(len(p2[n])):
                        if n_ in p2[n][i]:
                            self.Q.update({((2,i,n),(n_,n)): - delta3})
                
    def solver(self, Solver, num_reads=1, annealing_time=100):
        bqm = dimod.BinaryQuadraticModel.from_qubo(self.Q, self.s0_sum)
        if Solver == "Exact":
            self.response = ExactSolver().sample(bqm)
        elif Solver == "SA":
            self.response = neal.SimulatedAnnealingSampler().sample(bqm, num_reads=num_reads)
        elif Solver == "QBSolv":
            self.response = QBSolv().sample(bqm)
        elif Solver == "Quantum":
            self.response = EmbeddingComposite(DWaveSampler()).sample(bqm, annealing_time=annealing_time) 
    
    def visualization(self, num=0, visual=False):
        for i, r in enumerate(self.response):
            if i == num:
                sample = r
        edges = []
        const = 0
        for n in range(self.N):
            if self.p1[n] != []:
                tmp = 0
                for i in range(len(self.p1[n])):
                    if sample[(1,i,n)] == 1:
                        for m in list(self.p1[n][i]):                            
                            edges += [(m, n)]
                        tmp += 1
                if tmp > 1:
                    const += 1
            if self.p2[n] != []:
                tmp = 0
                for i in range(len(self.p2[n])):
                    if sample[(2,i,n)] == 1:
                        for m in list(self.p2[n][i]):                            
                            edges += [(m, n)]
                        tmp += 1
                if tmp > 1:
                    const += 1
        G = nx.DiGraph(edges)
        if const > 0:
            print("Constraints Error: p is more than 2.")
        if nx.is_directed_acyclic_graph(G) == False:            
            print("Constraints Error: Not DAG")
        score = self.s0_sum
        for n in range(len(self.s1)) :
            tmp = set()
            for e in edges:
                if e[1] == n:
                    tmp = tmp | {e[0]}
            u1 = set()
            for p1_ in self.p1[n]:
                u1 = u1 | p1_
            u2 = set()
            for p2_ in self.p2[n]:
                u2 = u2 | p2_
            for i, p1_ in enumerate(self.p1[n]):
                if p1_ == tmp & u1:
                    score += self.s1[n][i]
            for j, p2_ in enumerate(self.p2[n]):
                if p2_ == tmp & u2:
                    score += self.s2[n][j]
            for i, p1_ in enumerate(self.p1[n]):
                for j, p2_ in enumerate(self.p2[n]):
                    if p1_ | p2_ == tmp:
                        score += self.t[n][i][j]
        if visual != False:
            print('score:', score)
            pylab.figure()
            pos = nx.spring_layout(G)
            nx.draw_networkx_nodes(G, pos, node_size=8, node_color="w")
            nx.draw_networkx_edges(G, pos, width=1)
            nx.draw_networkx_labels(G, pos, font_size=8, font_color="r")
            pylab.savefig('./Graph.eps')
            pylab.show()
        return score, G