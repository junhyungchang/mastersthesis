# Code descriptions for Master's thesis project

In order to run all three methods, run GPregression.m.

* Note 0: running GPregression.m requires pulling the entire repository.

* Note 1: GPregression.m will take a while to run completely, but will print intermediate time stamps.

* Note 2: the Chebfun package, and Symbolic math toolbox is required to run GPregression.m.
https://www.chebfun.org/

* Note 3: For the code hyperMLE.m, Matlab's optimization toolbox is also required (fminunc).


## Routines

### Low-rank approximations

* REig.m: function for global low-rank factors via single-pass  Hermitian Randomized SVD.

* RLR.m: function that returns low-rank factors U,V such that
	A \approx U*V.
	Uses randomized method
	input is a matrix A, and desired rank r and over-sampling parameter p.
	(typically r=15, p=10).
	Call is [U,V] = PQR(A, r, p)

* KLLR.m: Low rank factors via KL-expansion for off-diagonal low-rank matrices.

* PQR.m: function that returns low-rank factors U,V such that
	A \approx U*V.
	Uses deterministic method (rank-revealing pivoted qr)
	input is a matrix A, and desired rank r (typically r=15).
	Call is [U,V] = PQR(A, r)

* IDQR: Interpolative decomposition via economy sized QR factorization.

* ACA.m: Code for adaptive cross approximation. 


### Hierarchical factorizations

* HODLR.m: function for posterior mean via HODLR factorization 

* RSFapply.m: computes the recursive skeletonization factorization.

* peeling.m: Matrix peeling that returns the trace of a black-box matrix.


### Karhunen-Loeve expansion
* KLexpansion.m: function for global low-rank factors via KL-expansion 
		(requires Chebfun and Symbolic math toolbox)

### Kernels
* sqexp.m: function for squared exponential kernel.

* sqexpdiff.m: derivative for sqexp.m used for hyperparameter MLE.

* matern.m: Matern kernel.

* exponential.m: exponential kernel.

* rquadratic.m: Rational quadratic kernel.

### k-d tree sorting

* treesort.m: Takes a nxd data matrix, and k as input, and performs a perfect k-d tree sorting of the input data.









