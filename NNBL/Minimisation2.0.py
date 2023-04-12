#!/usr/bin/env python
# coding: utf-8

# # Minimization Algorithm 2.0

# In[61]:


import numpy as np


# In[2]:


def updateA(X, S):
    n, l = X.shape
    regularizer = 0.00001
    I = np.eye(n)
    
    XST = X @ S.T
    SST = S @ S.T + regularizer * I
    A = XST @ np.linalg.pinv(SST)
    return A


# In[4]:


def updateS(X, A, b, m):  
    regularizer = 0.00001
    ATA = A.T @ A
    bbT = b @ b.T
    n, _ = ATA.shape 
    I = np.eye(n)
    F1 = ATA + bbT + regularizer * I
    bm = b @ m
    ATX = A.T @ X
    F2 = bm + ATX
    S = np.linalg.inv(F1) @ F2
    return S


# In[23]:


def updateb(S, m):
    n, l = S.shape
    regularizer = 0.00001
    I = np.eye(n)
    SST = S @ S.T + regularizer * I
    SmT = S @ m.T
    b = np.linalg.pinv(SST) @ SmT
    return b


# In[24]:


def projection(A, gamma1, gamma2):
    n, n = A.shape
    for i in range(n):
        for j in range(n):
            if i != j:
                if A[i][j] > gamma2:
                    A[i][j] = gamma2
                if A[i][j] < gamma1:
                    A[i][j] = gamma1
            if i == j:
                if A[i][j] > 1:
                    A[i][j] = 1
                if A[i][j] < 0.6:
                    A[i][j] = 0.6
    return A


# In[286]:


def projectb(b):
    n, _ = b.shape
    for i in range(n):
        if b[i] > 2:
            b[i] = 2
        if b[i] < 0.4:
            b[i] = 0.4
    return b


# In[28]:


def objective(X, A, S, b, m):
    return np.linalg.norm(X - A @ S)**2 + np.linalg.norm(m - b.T @ S)**2


# In[324]:


def minimise(X, A, S, m, gamma1, gamma2, max_iter=1000, tol=1e-12):
    
    n, l = X.shape
    #A_opt = np.eye(n)
    #'''
    A_opt = np.array([[1, 0.2, 0.01],
              [0.2, 0.92, 0.15],
              [0.05, 0.21, 0.8]])
    A_opt = A_opt + 0.05 * np.ones(n)
    #'''
    S_opt = X.copy()
    b_opt = np.ones((3, 1))
    
    iters = 0
    while True:
        
        A_opt = updateA(X, S_opt)
        A_opt = projection(A_opt, gamma1, gamma2)
        
        S_opt = updateS(X, A_opt, b_opt, m)
        
        b_opt = updateb(S_opt, m)
        b_opt = projectb(b_opt)
        
        error = objective(X, A_opt, S_opt, b_opt, m)
        
        if (iters > 0 and (prev_error - error) <= tol) or iters > max_iter:
            break
        
        prev_error = error
        
        print('Iteration:', iters+1, 'ERROR:', error, sep=' ')
        
        iters += 1
    
    return A_opt, S_opt, b_opt


# In[ ]:





# In[325]:


S = np.random.random((3, 10))
A = np.array([[1, 0.2, 0.01],
              [0.2, 0.92, 0.15],
              [0.05, 0.21, 0.8]])
X = A @ S
b = np.array([[1, 1, 1]]).T
m = b.T @ S


# In[331]:


gamma1 = 0.0001
gamma2 = 0.3
A_pred, S_pred, b_pred = minimise(X, A, S, m, gamma1, gamma2)
A_pred, b_pred


# In[ ]:





# In[ ]:





# In[ ]:





# # Gradient Descent

# In[356]:


def UpdateA(X, A, S, eta):
    X_AST = (X - (A @ S)).T
    gradA = S @ X_AST
    A = A + eta * gradA
    return A


# In[357]:


def UpdateS(X, A, S, m, b, eta):
    ATX_AS = A.T @ (X - (A @ S))
    bbTS_bm = (b @ b.T @ S) - (b @ m)
    gradS = ATX_AS - bbTS_bm
    S = S + eta * gradS
    return S


# In[371]:


def Updateb(S, m, b, eta):
    gradb = S @ (m.T - (S.T @ b))
    b = b + eta * gradb
    return b


# In[372]:


def projectb(b):
    n, _ = b.shape
    for i in range(n):
        if b[i] > 2:
            b[i] = 1
        if b[i] < 0.4:
            b[i] = 0.4
    return b


# In[377]:


def GradDescentMinimise(X, A, S, m, gamma1, gamma2, eta=0.05, max_iter=1000, tol=1e-6):
    
    n, l = X.shape
    #A_opt = np.eye(n)
    #'''
    A_opt = np.array([[1, 0.2, 0.01],
              [0.2, 0.92, 0.15],
              [0.05, 0.21, 0.8]])
    A_opt = A_opt + 0.05 * np.ones(n)
    #'''
    S_opt = X.copy()
    b_opt = np.ones((3, 1))
    
    iters = 0
    while True:
        
        A_opt = UpdateA(X, A_opt, S_opt, eta)
        A_opt = projection(A_opt, gamma1, gamma2)
        
        S_opt = UpdateS(X, A_opt, S_opt, m, b_opt, eta)
        
        b_opt = Updateb(S_opt, m, b_opt, eta)
        b_opt = projectb(b_opt)
        
        error = objective(X, A_opt, S_opt, b_opt, m)
        
        if (iters > 0 and (prev_error - error) <= tol) or iters > max_iter:
            break
        
        prev_error = error
        
        print('Iteration:', iters+1, 'ERROR:', error, sep=' ')
        
        iters += 1
    
    return A_opt, S_opt, b_opt


# In[378]:


S = np.random.random((3, 10))
A = np.array([[1, 0.2, 0.01],
              [0.2, 0.92, 0.15],
              [0.05, 0.21, 0.8]])
X = A @ S
b = np.array([[1, 1, 1]]).T
m = b.T @ S


# In[379]:


gamma1 = 0.0001
gamma2 = 0.5
eta = 0.05
A_pred, S_pred, b_pred = GradDescentMinimise(X, A, S, m, gamma1, gamma2, eta=0.1, tol=1e-6)


# In[380]:


A_pred, b_pred


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




