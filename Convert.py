###############################################################################
# Conversion
###############################################################################

from itertools import chain,combinations

class Conversion:
    
    def __init__(self, z_max=0):
        self.z_max = z_max    

    def score(self, U):
        for i, U_ in enumerate(self.U_list):
            if U_ <= U <= U_.union(self.V_list[i]):
                return self.Score_list[i]
    
    def Uset(self, U):
        Uset = []
        for U_ in U:
            if U_ not in Uset:
                Uset += [U_]
        return Uset
    
    def fit(self, U_list, V_list, Score_list):
        self.U_list = U_list
        self.V_list = V_list
        self.Score_list = Score_list
        Ucup = set()
        for U in U_list:
            Ucup = Ucup.union(U)
        Z_list = list(chain.from_iterable(combinations(Ucup, r) 
                        for r in range(min(int(len(Ucup)/2)+1,self.z_max+1))))
        bits = len(U_list) - 1
        for Z in Z_list:
            U1 = [U & set(Z) for U in U_list] 
            U1 = self.Uset(U1)
            U2 = [U & (Ucup - set(Z)) for U in U_list] 
            U2 = self.Uset(U2)
            U12 = [U1_ | U2_ for U1_ in U1 for U2_ in U2]
            if sum([1 for U in U_list if U in U12]) == len(U_list):
                bits_ = len(U1) + len(U2) - 2
                if bits_ <= bits:
                    bits = bits_
                    self.p1 = U1
                    self.p2 = U2
        self.p1 = [p1_ for p1_ in self.p1 if p1_ != set()]
        self.p2 = [p2_ for p2_ in self.p2 if p2_ != set()]
        self.s0 = self.score(set())
        self.s1 = [self.score(p1_) - self.s0 for p1_ in self.p1]
        self.s2 = [self.score(p2_) - self.s0 for p2_ in self.p2]
        self.t = [[self.score(p1_ | p2_) - self.score(p1_) - self.score(p2_) 
                 + self.s0 for p2_ in self.p2] for p1_ in self.p1]