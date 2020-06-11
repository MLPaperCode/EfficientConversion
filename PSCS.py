###############################################################################
# Parent Set Candidates Selection
###############################################################################

import numpy as np
from sklearn import tree
from sklearn.preprocessing import LabelBinarizer
import copy

class ParentSetCandidatesSelection:
    
    def __init__(self, D, min_impurity_decrease):
        self.D = np.array(D)
        self.min_impurity_decrease = min_impurity_decrease
        self.M = D.shape[1]
        self.X = set([m for m in range(self.M)])
        
    def Pa(self, W):
        Dw = copy.copy(self.D)
        for m in range(self.M):
            if m not in W:
                Dw[:, m] = 1
        clf = tree.DecisionTreeClassifier(criterion = "entropy",
                            min_impurity_decrease = self.min_impurity_decrease)
        clf.fit(Dw, self.Dn)
        U = set([m for m in set(clf.tree_.feature) if m >= 0])
        prob = clf.predict_proba(Dw)
        e = 0.00001
        if self.State.shape[1] == 2:
            Score = sum([sum([-self.State[i,s]*np.log(min(max(prob[i,s],e),1-e)) 
                        for s in range(2)]) for i in range(self.Dn.shape[0])])
        else:
            Score = sum([sum([-self.State[i,s]*np.log(min(max(prob[s][i,1],e),1-e)) 
                for s in range(self.Dn.shape[1])]) for i in range(self.Dn.shape[0])])            
        return U, Score
        
    def check_list(self, W):
        for i, U_ in enumerate(self.U_list):
            if U_ <= W <= U_.union(self.V_list[i]):
                return self.Score_list[i]
        return False
    
    def RecursiveTree(self, W):
        U, Score = self.Pa(W)
        V = W - U
        self.U_list += [U]
        self.V_list += [V]
        self.Score_list += [Score]
        if U != set():
            for X_ in U:
                if self.check_list(W - {X_}) == False:
                    self.RecursiveTree(W - {X_})
    
    def fit(self, n):
        self.U_list = []
        self.V_list = []
        self.Score_list = []
        self.Dn = LabelBinarizer().fit_transform(self.D[:,n])   
        if self.Dn.shape[1] == 1:
            self.State = np.array([[1 - self.Dn[i,0], self.Dn[i,0]] 
                                      for i in range(len(self.Dn))])
        else:
            self.State = copy.copy(self.Dn)
        self.RecursiveTree(self.X - {n})  