import numpy as np


class minimise:
	
    def __init__(self):
        return 0
	
    def updateA(self, X, S):
        n, l = X.shape
        I = np.eye(n)
        regularizer = 0.00001
        XST = X @ S.T
        SST = S @ S.T
        F1 = (XST + regularizer*I)
        F2 = (SST + regularizer*I)
        F2_inv = np.linalg.inv(F2)
        A = F1 @ F2_inv
        return A

    def updateS(self, X, A, m):
        n, l = X.shape
        O = np.ones((n, 1))
        ATA = A.T @ A
        ATX = A.T @ X
        F1 = (ATA + O @ O.T)
        F1_inv = np.linalg.inv(F1)
        F2 = (O @ m + A.T @ X)
        S = F1_inv @ F2
        return S

    def objective(self, X, A, S, m):
        return np.linalg.norm(X - (A @ S)) ** 2 + np.linalg.norm(m - np.sum(S, axis=0)) ** 2


    def project(self, A, gamma1, gamma2):
        n, _ = A.shape
        for i in range(n):
            for j in range(n):
                if i != j:
                    if A[i][j] > gamma2:
                        A[i][j] = gamma2
                    if A[i][j] < gamma1:
                        A[i][j] = gamma1
                else:
                    A[i][j] = 1
        return A


    def factor(self, X, m, gamma1, gamma2, tol=1e-3, max_iter=1000):
        n, l = X.shape
        A = np.eye(n)
        S_opt = X.copy()

        iters = 0
        while True:

            A_opt = self.updateA(self, X, S_opt)
            A_opt = self.project(self, A_opt, gamma1, gamma2)
            S_opt = self.updateS(self, X, A_opt, m)
            loss = self.objective(self, X, A_opt, S_opt, m)
            # print('ITERATION:', iters+1, 'LOSS:', loss, sep=' ')
            if iters > 0 and (prev_loss - loss) <= tol:
                break
            prev_loss = loss
            iters += 1
            if iters >= max_iter:
                print('MAXIMUM ITERATION REACHED!!!')
                break

        return A_opt, S_opt
