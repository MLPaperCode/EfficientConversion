###############################################################################
# Verification
###############################################################################
data = "Real" # "Real", "Pseudo"

# Real
infile = 'application_train.csv' 
S = 3 # the maximum number of state
min_impurity_decrease = 0.05 # threshold
z_max = 3 # maximum size of subset

# Pseudo
N_sub = 5 # small graph size
N_G = 20 # the number of small graphs
Lambda_max = 5 # the maximum number of parent set candidates

# Solver
num_reads = 1000 # samples
Solver = "SA" # solver type
###############################################################################

import numpy as np
import pandas as pd 
import time 
from Preprocess import Preprocessing
from PSCS import ParentSetCandidatesSelection 
from Convert import Conversion
from Pseudo import PseudoDataset
from QUBO_Solver import QUBOSolver

if data == "Real":
    D = pd.read_csv(infile)
    D = D.iloc[:,1:]
    D = Preprocessing(drop=False).transform(D, S)
    p1 = [] # [N][Lambda1]
    p2 = [] # [N][Lambda2]
    s0 = [] # [N]
    s1 = [] # [N][Lambda1]
    s2 = [] # [N][Lambda2]
    t = [] # [N][Lambda1][Lambda2]
    delta_lower = 0 
    PSCS = ParentSetCandidatesSelection(D, min_impurity_decrease)
    Conv = Conversion(z_max=z_max)
    for n in range(D.shape[1]):
        time0 = time.time()
        PSCS.fit(n)
        PSCS_time = time.time() - time0
        Conv.fit(PSCS.U_list, PSCS.V_list, PSCS.Score_list)
        p1 += [Conv.p1]
        p2 += [Conv.p2]
        s0 += [Conv.s0]
        s1 += [Conv.s1]
        s2 += [Conv.s2]
        t += [Conv.t]
        Omega_1 = len(PSCS.U_list) - 1
        Lambda_1 = len(Conv.Uset(PSCS.U_list)) - 1
        Lambda12_2 = len(Conv.p1) + len(Conv.p2)
        if Conv.s1 == [] and Conv.s2 == [] :
            delta_lower = delta_lower
        elif Conv.s1 == []:
            delta_lower = max(delta_lower, -min(Conv.s2))
        elif Conv.s2 == []:
            delta_lower = max(delta_lower, -min(Conv.s1))
        else:
            delta_lower = max(delta_lower, 
                          max([-Conv.s1[i]-min(np.array(Conv.t)[i,:]) for i in range(len(Conv.s1))]),
                          max([-Conv.s2[j]-min(np.array(Conv.t)[:,j]) for j in range(len(Conv.s2))]))
        print("n:", n, "time: {:.2f}".format(PSCS_time), "(sec)", 
              "Omega-1:", Omega_1, "Lambda-1:", Lambda_1, "Lambda1+Lambda2-2:", Lambda12_2)
    
elif data == "Pseudo":
    PD = PseudoDataset(N_sub=N_sub, Lambda_max=Lambda_max, N_G=N_G)
    p1, p2, s0, s1, s2, t, delta_lower, score, G0 = PD.generate()
    print('minimum score:', score)

QUBOS = QUBOSolver()
QUBOS.Hamiltonian(p1, p2, s0, s1, s2, t, delta_lower) 
QUBOS.solver(Solver, num_reads=num_reads)
score_, G_ = QUBOS.visualization(visual=True)