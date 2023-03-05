import math
import numpy.linalg

class RPCA:

    def robust_pca(self, M):
        """ 
        Decompose a matrix into low rank and sparse components.
        Computes the RPCA decomposition using Alternating Lagrangian Multipliers.
        Returns L,S the low rank and sparse components respectively
        """
        L = numpy.zeros(M.shape)
        S = numpy.zeros(M.shape)
        Y = numpy.zeros(M.shape)
        print(M.shape)
        mu = (M.shape[0] * M.shape[1]) / (4.0 * self.L1Norm(self, M))
        lamb = max(M.shape) ** -0.5
        while not self.converged(self, M,L,S):
            L = self.svd_shrink(self, M - S - (mu**-1) * Y, mu)
            S = self.shrink(self, M - L + (mu**-1) * Y, lamb * mu)
            Y = Y + mu * (M - L - S)

        self.L = L
        self.S = S
        self.M = M
        return L,S
        
    def svd_shrink(self, X, tau):
        """
        Apply the shrinkage operator to the singular values obtained from the SVD of X.
        The parameter tau is used as the scaling parameter to the shrink function.
        Returns the matrix obtained by computing U * shrink(s) * V where 
            U are the left singular vectors of X
            V are the right singular vectors of X
            s are the singular values as a diagonal matrix
        """
        U,s,V = numpy.linalg.svd(X, full_matrices=False)
        return numpy.dot(U, numpy.dot(numpy.diag(self.shrink(self, s, tau)), V))
        
    def shrink(self, X, tau):
        """
        Apply the shrinkage operator the the elements of X.
        Returns V such that V[i,j] = max(abs(X[i,j]) - tau,0).
        """
        V = numpy.copy(X).reshape(X.size)
        for i in range(V.size):
            V[i] = math.copysign(max(abs(V[i]) - tau, 0), V[i])
            if V[i] == -0:
                V[i] = 0
        return V.reshape(X.shape)
                
    def frobeniusNorm(self, X):
        """
        Evaluate the Frobenius norm of X
        Returns sqrt(sum_i sum_j X[i,j] ^ 2)
        """
        accum = 0
        V = numpy.reshape(X,X.size)
        for i in range(V.size):
            accum += abs(V[i] ** 2)
        return math.sqrt(accum)

    def L1Norm(self, X):
        """
        Evaluate the L1 norm of X
        Returns the max over the sum of each column of X
        """
        return max(numpy.sum(X,axis=0))

    def converged(self, M,L,S):
        """
        A simple test of convergence based on accuracy of matrix reconstruction
        from sparse and low rank parts
        """
        error = self.frobeniusNorm(self, M - L - S) / self.frobeniusNorm(self, M)
        print ("error =", error)
        return error <= 10e-6