###############################################################################
infile = 'application_train.csv' #Input csv file 
###############################################################################

import numpy as np
import pandas as pd 

def Processed(D, S, drop=True):
    if drop == True:
        D = D.dropna()
    #D = D.fillna('Error')
    D = D.replace('', 'Error')
    col = np.array(D.columns)
    D = np.array(D)
    coldel = []
    for m in range(D.shape[1]):
        Dm = [D[i,m] for i in range(D.shape[0]) if D[i,m] != 'Error']
        tf = all([any([isinstance(Dm[i], float), isinstance(Dm[i], int)]) 
                                                for i in range(len(Dm))])
        Dm_ = list(set(Dm))
        if tf == True and len(Dm_) > S:
            th_ = [np.percentile(Dm, (s+1) / S * 100) for s in range(S)]
            for i in range(D.shape[0]):
                if D[i,m] != 'Error' :
                    for s, th in enumerate(th_):
                        if D[i,m] <= th:
                            D[i,m] = s + 1
                            break
                else:
                    D[i,m] = 0
        elif len(Dm_) <= 1 or len(Dm_) > S:
            coldel += [m]
        else:
            for i in range(D.shape[0]):
                for num, s in enumerate(Dm_):
                    if D[i,m] == s:
                        D[i,m] = num + 1
                        break
                if D[i,m] == 'Error' :
                    D[i,m] = 0
    col = list(np.delete(col, coldel))
    D = np.delete(D, coldel, 1)
    D = pd.DataFrame(np.array(D, dtype=np.int64))
    D.columns = col
    return D

D = pd.read_csv(infile)
D = D.iloc[:,1:]
S = 3
D = Processed(D, S)
D.to_csv('./HomeCredit_Processed.csv')