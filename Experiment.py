###############################################################################
Download = "ON" # "ON", "OFF"
Fig = "Fig3" # "Fig2", "Fig3"
###############################################################################

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import time
from Preprocess import Preprocessing
from PSCS import ParentSetCandidatesSelection 
from Convert import Conversion

if Download == "ON" :
    infile = 'application_train.csv' 
    S = 3
    D = pd.read_csv(infile)
    D = D.iloc[:,1:]
    D = Preprocessing(drop=False).transform(D, S)

if Fig == "Fig2":
    min_impurity_decrease_list = [0.075, 0.050, 0.025]
    figL = plt.figure()
    axL = figL.add_subplot()
    figR = plt.figure()
    axR = figR.add_subplot()
    color_list = ['r', 'g', 'b']
    marker_list = ['o', '^', 'x']
    save_list = ['./Fig2L.eps', './Fig2R.eps']
    time_list = [[] for i in range(len(min_impurity_decrease_list))]
    log2_Lambda_list = [[] for i in range(len(min_impurity_decrease_list))]
    Omega_list = [[] for i in range(len(min_impurity_decrease_list))]
    ratio_list = [[] for i in range(len(min_impurity_decrease_list))]
    for n in range(D.shape[1]):
        for num, min_impurity_decrease in enumerate(min_impurity_decrease_list):
            PSCS = ParentSetCandidatesSelection(D, min_impurity_decrease)        
            time0 = time.time()
            PSCS.fit(n)
            time1 = time.time() - time0
            time_list[num] += [time1]
            Lambda = len(Conversion().Uset(PSCS.U_list)) 
            log2_Lambda_list[num] += [np.log2(Lambda)]
            Omega = len(PSCS.U_list)
            Omega_list[num] += [Omega]
            ratio_list[num] += [Omega / Lambda]
            print("threshold:", min_impurity_decrease, "n:", n,                   
                  "Lambda:", Lambda, "Omega / Lambda:", Omega / Lambda,
                  "time: {:.2f}".format(time1), "(sec)")
    fig = plt.figure()           
    for num, min_impurity_decrease in enumerate(min_impurity_decrease_list):
        plt.plot(log2_Lambda_list[num], ratio_list[num], linestyle='None', 
                 color=color_list[num], marker=marker_list[num], 
                 label='threshold = ' + '{:.3f}'.format(min_impurity_decrease))    
    plt.legend()
    plt.xlabel('$\log_2 \Lambda_n$')
    plt.ylabel('$\Omega_n$ / $\Lambda_n$')
    plt.savefig('./Fig2L.eps')
    fig = plt.figure()
    for num, min_impurity_decrease in enumerate(min_impurity_decrease_list):
        plt.plot(Omega_list[num], time_list[num], linestyle='None', 
                 color=color_list[num], marker=marker_list[num], 
                 label='threshold = ' + '{:.3f}'.format(min_impurity_decrease))
    plt.legend()
    plt.xlabel('$\Omega_n$')
    plt.ylabel('Calculation Time [sec]')
    plt.savefig('./Fig2R.eps')
elif Fig == "Fig3":
    min_impurity_decrease = 0.05
    z_max_list = [1, 2, 3]
    color_list = ['c', 'm', 'y']
    marker_list = ['s', 'v', '+']
    Lambda_1_list = []    
    U_list = []    
    Lambda12_2_list = [[] for i in range(len(z_max_list))]
    PSCS = ParentSetCandidatesSelection(D, min_impurity_decrease)    
    for n in range(D.shape[1]):
        PSCS.fit(n)
        Lambda_1 = len(Conversion().Uset(PSCS.U_list)) - 1
        Lambda_1_list += [Lambda_1]
        U_union = set()
        for U in PSCS.U_list:
            U_union = U_union.union(U)
        U_list += [len(U_union)]    
        for num, z_max in enumerate(z_max_list):            
            Conv = Conversion(z_max=z_max)
            time0 = time.time()
            Conv.fit(PSCS.U_list, PSCS.V_list, PSCS.Score_list)
            time1 = time.time() - time0
            Lambda12_2 = len(Conv.p1) + len(Conv.p2)
            Lambda12_2_list[num] += [Lambda12_2]
            print("n:", n, "z_max", z_max,
                  "Lambda-1:", Lambda_1, "Lambda1+Lambda2-2:", Lambda12_2, 
                  "time: {:.2f}".format(time1), "(sec)")
    fig = plt.figure()
    for num, z_max in enumerate(z_max_list):
        plt.plot(Lambda_1_list, Lambda12_2_list[num], linestyle='None', 
                 color=color_list[num], marker=marker_list[num], 
                 label='k =' + str(z_max))
    plt.legend()
    plt.xlabel('$\Lambda_n - 1$')
    plt.ylabel('$\Lambda^{(k)} - 2$')
    plt.savefig('./Fig3L.eps')
    fig = plt.figure()
    plt.plot(Lambda_1_list, U_list, linestyle='None', color='k', marker='.')
    plt.legend()
    plt.xlabel('$\Lambda_n - 1$')
    plt.ylabel('$|\cup_{\lambda = 0}^{\Lambda_n - 1} U_{\lambda,n}|$')
    plt.savefig('./Fig3R.eps')