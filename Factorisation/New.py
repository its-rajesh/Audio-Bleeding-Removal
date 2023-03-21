#!/usr/bin/env python
# coding: utf-8

# In[125]:


import numpy as np
from scipy.optimize import minimize


# In[128]:


def objective(params, X, m, gamma1, gamma2):
    A = params[:n**2].reshape((n, n))
    S = params[n**2:].reshape((n, l))
    t = np.sum(S, axis=0)
    return np.linalg.norm(X - np.dot(A, S))**2 + np.linalg.norm(m - t)**2

def update_A(S, X, gamma2):
    E = np.eye(n) - 1/n
    A = np.dot(X, S.T) + gamma2 * E
    SS = np.dot(S, S.T) + gamma2 * np.eye(l)
    A = np.dot(A, np.linalg.inv(SS))
    np.fill_diagonal(A, 1)
    return A.ravel()

def update_S(A, X):
    return np.linalg.lstsq(A, X, rcond=None)[0].ravel()

def minimize_function(X, m, gamma1, gamma2):

    # initialize A as identity matrix and S as X
    n, l = X.shape
    A_init = np.eye(n)
    S_init = X.copy()

    # set bounds for A and S separately
    A_bounds = [(1, 1) if i == j else (gamma1, gamma2) for i in range(n) for j in range(n)]
    S_bounds = [(-np.inf, np.inf)] * (n * l)
    bounds = A_bounds + S_bounds

    # alternating optimization
    params = np.concatenate([A_init.ravel(), S_init.ravel()])
    res = minimize(objective, params, args=(X, m, gamma1, gamma2), method='L-BFGS-B', bounds=bounds)

    # extract the optimized A and S
    A_opt = res.x[:n**2].reshape((n, n))
    S_opt = res.x[n**2:].reshape((n, l))
    
    return A_opt, S_opt


# In[160]:


A = np.array([[1, 0.02, 0.22],
             [0.1, 1, 0.02],
             [0.1, 0.1, 1]])


# In[161]:


S = np.random.random((3, 5))
X = A @ S


# In[162]:


m = S[0]+S[1]+S[2]


# In[167]:


gamma1 = 0
gamma2 = 1
A_opt, S_opt = minimize_function(X, m, gamma1, gamma2)
l1 = np.linalg.norm(X - np.dot(A_opt, S_opt))
l2 = np.linalg.norm(m - (S_opt[0]+S_opt[1]+S_opt[2]))
X, A_opt, S_opt, S, l1, l2, l1+l2


# In[168]:


np.linalg.norm(A-A_opt), np.linalg.norm(S-S_opt)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




