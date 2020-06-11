###############################################################################
# Preprocessing
###############################################################################

import numpy as np
import pandas as pd 

class Preprocessing():
    
    def __init__(self, drop=True):
        self.drop = drop
    
    def transform(self, D, S):
        if self.drop == True:
            D = D.dropna()
        else:
            D = D.fillna('Error')
        D = D.replace('', 'Error')
        col = np.array(D.columns)
        D = np.array(D)
        coldel = []
        for m in range(D.shape[1]):
            Dm = [D[i,m] for i in range(D.shape[0]) if D[i,m] != 'Error']
            tf = any([isinstance(Dm[i], str) for i in range(len(Dm))])
            Dm_ = list(set(Dm))
            if tf == False and len(Dm_) > S:
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
