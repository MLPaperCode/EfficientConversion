Name: Efficient Conversion of Bayesian Network Learning into Quadratic Unconstrained Binary Optimization

Language:Python

Requirement:
scikit-learn (pip install scikit-learn), 
networkx (pip install networkx), 
ocean (pip install dwave-ocean-sdk). 

Contents:
Experiment.py (Reproduce the experimental results)
Verification.py (Verificate our conversion using two type of dataset)
※ Download application_train.csv from https://www.kaggle.com/c/home-credit-default-risk/data

Modules:
Preprocess.py
PSCS.py
Convert.py
Pseudo.py
QUBO_Solver.py  

Pseudo Dataset:
\begin{algorithm}[h]
   \caption{Pseudo Dataset}
   \label{Pseudo}
\begin{algorithmic}[1]
   \State Create multiple of small exact BNs, and rebuild them as one exact BN $\mathcal{G}^*$. 
   \Repeat
   \State Pick up one variable $X_n$ which have no parent set candidates except for $U_{0,n} = \phi$. 
   \State Set some parent set candidates $(U_{\lambda,n})_{\lambda=1}^{\Lambda_n}$ of $X_n$ and each score $(S_n(U_{\lambda,n}))_{\lambda=1}^{\Lambda_n}$ randomly. 
   \State Chose the parent set candidate with minimum score, and add it to $\mathcal{G}^*$. 
   \State If the graph is DAG, then renewal the parent set candidates and the exact BN $\mathcal{G}^*$. 
   \Until{There is no renewal.}
\end{algorithmic}
\end{algorithm}