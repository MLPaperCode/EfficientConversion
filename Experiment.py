###############################################################################
Fig = "Fig3" # "Fig2", "Fig3"
###############################################################################

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from PSCS import ParentSetCandidatesSelection 
from Convert import Conversion

infile = 'HomeCredit_Processed.csv' #Input csv file 
D = np.array(pd.read_csv(infile))
D = D[:,1:]

if Fig == "Fig2":
    min_impurity_decrease_list = [0.075, 0.050, 0.025]
    fig, ax = plt.subplots()
    color_list = ['r', 'g', 'b']
    marker_list = ['o', '^', 'x']
    for num, min_impurity_decrease in enumerate(min_impurity_decrease_list):
        log2_Lambda_list = []
        ratio_list = []
        PSCS = ParentSetCandidatesSelection(D, min_impurity_decrease)        
        for n in range(D.shape[1]):
            PSCS.fit(n)
            Lambda = len(Conversion().Uset(PSCS.U_list)) 
            log2_Lambda_list += [np.log2(Lambda)]
            Omega = len(PSCS.U_list)
            ratio_list += [Omega / Lambda]
            print("threshold:", min_impurity_decrease, "n:", n, 
                  "Lambda:", Lambda, "Omega / Lambda:", Omega / Lambda)
        ax.plot(log2_Lambda_list, ratio_list, linestyle='None', 
                color=color_list[num], marker=marker_list[num], 
                label='threshold = ' + '{:.3f}'.format(min_impurity_decrease))
    plt.legend()
    plt.xlabel('$\log_2 \Lambda_n$')
    plt.ylabel('$\Omega_n$ / $\Lambda_n$')
    plt.savefig('./Fig2.eps')
    plt.show()
    
elif Fig == "Fig3":
    min_impurity_decrease = 0.025
    PSCS = ParentSetCandidatesSelection(D, min_impurity_decrease)    
    z_max_list = [0, 2, 4]
    fig, ax = plt.subplots()
    color_list = ['r', 'g', 'b']
    marker_list = ['o', '^', 'x']    
    Lambda_1_list = []
    Lambda12_2_list = [[] for i in range(len(z_max_list))]
    for n in range(D.shape[1]):
        for num, z_max in enumerate(z_max_list):
            PSCS.fit(n)
            Conv = Conversion(z_max=z_max)
            Conv.fit(PSCS.U_list, PSCS.V_list, PSCS.Score_list)
            Lambda_1 = len(Conv.Uset(PSCS.U_list)) - 1
            Lambda_1_list += [Lambda_1]
            Lambda12_2 = len(Conv.p1) + len(Conv.p2)
            Lambda12_2_list[num] += [Lambda12_2]
            print("n:", n, "z_max", z_max, "Lambda-1:", Lambda_1, "Lambda1+Lambda2-2:", Lambda12_2)
    for num, z_max in enumerate(z_max_list):
        ax.plot(Lambda_1_list, Lambda12_2_list[num], linestyle='None', 
                color=color_list[num], marker=marker_list[num], label='k =' + str(z_max))
    plt.legend()
    plt.xlabel('$\Lambda_n - 1$')
    plt.ylabel('$\Lambda^{(k)} - 2$')
    plt.savefig('./Fig3.eps')
    plt.show()