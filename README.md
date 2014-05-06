SVDIFP
======

An efficient algorithm for computing a few extreme singular values of large sparse matrices(SVD)


SVDIFP.m is a MATLAB program that computes a few smallest or largest singular values of a large m by n matrix A:

      A u = sigma v and A' v = sigma u

where A is an m by n matrix, sigma is a singular value, u a corresponding right singular vector, and v a corresponding left singular vector. It computes the singular values of A through computing the eigenvalues of A'A using the inverse free preconditioned Krylov subspace method (EIGIFP). The basic SVDIFP algorithm is described in

Q. Liang and Q. Ye, Computing Singular Values of Large Matrices With Inverse Free Preconditioned Krylov Subspace Method.
The program is intended for efficiently computing a few clustered extreme singular values through preconditioning.

You need to download the main routine SVDIFP.m. You may also want to download RIF.c, which constructs the robust incomplete factorization (RIF) for preconditioning (see M. Benzi and M. Tuma, A Robust Preconditioner with Low Memory Requirements for Large Sparse Least Squares Problems, SIAM J. Scientific Computing, 25 (2003), 499-512). The construction of the preconditioner is the most time consuming part of the method. Both MATLAB implementation (within svdifp.m) and C implementation are provided. To use RIF.c, invoke MATLAB's mex functionality by

      mex -largeArrayDims RIF.c

Our RIF routines are based on the implementation of rifnrimex.m by Benzi and Tuma. When an RIF executable file is not present, the built-in MATLAB implementation will be used.

You may also find the information of this project via: https://www.ms.uky.edu/~qye/svdifp.html
