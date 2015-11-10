# LowRankApprox

[![Build Status](https://travis-ci.org/klho/LowRankApprox.jl.svg?branch=master)](https://travis-ci.org/klho/LowRankApprox.jl)

This Julia package provides fast low-rank approximation algorithms for BLAS/LAPACK-compatible matrices based on some of the latest technology in adaptive randomized matrix sketching. Currently implemented algorithms include:

- sketch methods:
  - random Gaussian
  - random subset
  - subsampled random Fourier transform
  - sparse random Gaussian
- partial range finder
- partial factorizations:
  - QR decomposition
  - interpolative decomposition
  - singular value decomposition
  - Hermitian eigendecomposition
  - CUR decomposition
- spectral norm estimation

By "partial", we mean essentially that these algorithms are early-terminating, i.e., they are not simply post-truncated versions of their standard counterparts. There is also support for "matrix-free" linear operators described only through their action on vectors. All methods accept a number of options specifying, e.g., the rank, estimated absolute precision, and estimated relative precision of approximation.

Our implementation borrows heavily from the perspective espoused by [N. Halko, P.G. Martinsson, J.A. Tropp. Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. SIAM Rev. 53 (2): 217-288, 2011.](http://dx.doi.org/10.1137/090771806), except that we choose the interpolative decomposition (ID) as our basic form of approximation instead of matrix range projection. The reason is that the latter requires expensive matrix-matrix multiplication to contract to the relevant subspace, while the former can sometimes be computed much faster, depending on the accelerated sketching strategy employed.

This package has been developed with performance in mind, and early tests have shown large speedups over similar codes written in MATLAB and Python (and even some in Fortran and C). For example, computing an ID of a Hilbert matrix of order 1024 to relative precision ~1e-15 takes:

- ~0.02 s using LowRankApprox in Julia
- ~0.07 s using SciPy in Python (calling a Fortran backend; see [PyMatrixID](http://klho.github.io/PyMatrixID))
- ~0.3 s in MATLAB

This difference can be attributed to Julia's tight integration with Fortran and C as well as to some low-level memory management not available in traditional dynamic languages.

LowRankApprox has been fully tested in Julia v0.4.0. The apparent build errors reported by Travis CI seem to be due to the reference 64-bit Julia installation being built with 32-bit BLAS.

## Installation

To install LowRankApprox, simply type:

```julia
Pkg.clone("git://github.com/klho/LowRankApprox.jl.git")
```

at the Julia prompt. The package can then be imported as usual with `using` or `import`.

## Getting Started

To illustrate the usage of this package, let's consider the computation of a partial QR decomposition of a Hilbert matrix, which is well known to have low rank. First, we load LowRankApprox via:

```julia
using LowRankApprox
```

Then we construct a Hilbert matrix with:

```julia
n = 1024
A = matrixlib(:hilb, n, n)
```

A partial QR decomposition can then be computed using:

```julia
F = pqrfact(A)
```

This returns a `PartialQR` factorization with variables `Q`, `R`, and `p` denoting the unitary, triangular, and permutation factors, respectively, constituting the decomposition. Alternatively, these can be extracted directly with:

```julia
Q, R, p = pqr(A)
```

but the factorized form is often more convenient as it also implements various arithmetic operations. For example, the commands:

```julia
x = rand(n)
y = F *x
z = F'*x
```

automatically invoke specialized multiplication routines to rapidly compute `y` and `z` using the low-rank structure of `F`.

The rank of the factorization can be retrieved by `F[:k]`, which in this example usually gives 26 or 27. The reason for this variability is that the default interface uses randomized Gaussian sketching for acceleration. Likewise, the actual approximation error is also random but can be efficiently and robustly estimated:

```julia
aerr = snormdiff(A, F)
rerr = aerr / snorm(A)
```

This computes the absolute and relative errors in the spectral norm using power iteration. Here, the relative error achieved should be on the order of machine epsilon. (You may see a warning about exceeding the maximum number of iterations, but this is harmless in this case.)

The default interface requests ~1e-15 estimated relative precision. To request, say, only 1e-12 relative precision, use:

```julia
rtol = 1e-12
F = pqrfact(A, rtol)
```

which returns a factorization of rank ~22. We can also directly control the rank instead with, e.g.:

```julia
rank = 20
F = pqrfact(A, rank)
```

Both of these are variants of a single unified interface of the form:

```julia
F = pqrfact(A, rank_or_rtol)
```

which interprets `rank_or_rtol` as a relative precision if `rank_or_rtol < 1` and as a rank otherwise.

The most general accuracy setting considers both the relative precision and the rank together, in addition to the absolute precision. For example, the code:

```julia
opts = LRAOptions(atol=1e-9, rank=20, rtol=1e-12)
F = pqrfact(A, opts)
```

sets three separate termination criteria: one on achieving estimated absolute precision 1e-9, another on achieving estimated relative precision 1e-12, and the last on reaching rank 20---with the computation completing upon any of these being fulfilled. Many other settings can also be specified through this "options" interface; we will discuss this in more detail later.

All of the above considerations also apply when the input is a linear operator, i.e., when the matrix is described not by its entries but by its action on vectors. To demonstrate, we can convert `A` to type `LinearOperator` as follows:

```julia
L = LinearOperator(A)
```

which inherits and stores methods for applying the matrix and its adjoint. (This command actually recognizes `A` as Hermitian and forms `L` as a `HermitianLinearOperator`.) A partial QR decomposition can then be computed with:

```julia
F = pqrfact(L)
```

just as in the previous case. Of course, there is no real benefit to doing this in this particular example; the advantage comes when considering complicated matrix products that can be represented implicitly as a single `LinearOperator`. For instance, `A*A` can be represented as `L*L` without ever forming the resultant matrix explicitly, and we can even encapsulate entire factorizations as linear operators to exploit fast multiplication:

```julia
L = LinearOperator(F)
```

Linear operators can be scaled, added, and composed together using the usual syntax. All methods in LowRankApprox transparently support both matrices and linear operators.

## Low-Rank Factorizations

We now detail the various low-rank approximations implemented, which all nominally return compact `Factorization` types storing the matrix factors in structured form. All such factorizations provide optimized multiplication routines. Furthermore, the rank of any factorization `F` can be queried with `F[:k]` and the matrix approximant defined by `F` can be reconstructed as `full(F)`. For concreteness of exposition, assume in the following that `A` has size `m` by `n` with factorization rank `F[:k] = k`.

### QR Decomposition

A partial QR decomposition is a factorization `A[:,p] = Q*R`, where `Q` is `m` by `k` with orthonormal columns, `R` is `k` by `n` and upper trapezoidal, and `p` is a permutation vector. Such a decomposition can be computed with:

```julia
F = pqrfact(A, args...)
```

or more explicitly with:

```julia
Q, R, p = pqr(A, args...)
```

The former returns a `PartialQR` factorization with access methods:

- `F[:Q]`: `Q` factor as type `Matrix`
- `F[:R]`: `R` factor as type `UpperTrapezoidal`
- `F[:p]`: `p` permutation as type `Vector`
- `F[:P]`: `p` permutation as type `ColumnPermutation`

Both `F[:R]` and `F[:P]` are represented as structured matrices, complete with their own arithmetic operations, and together permit the alternate approximation formula `A*F[:P] = F[:Q]*F[:R]`. The factorization form additionally supports least squares solution by left-division.

We can also compute a partial QR decomposition of `A'` (that is, pivoting on rows instead of columns) without constructing the matrix transpose explicitly by writing:

```julia
F = pqrfact(:c, A, args...)
```

and similarly with `pqr`. The default interface is equivalent to, e.g.:

```julia
F = pqrfact(:n, A, args...)
```

for "no transpose". It is also possible to generate only a subset of the partial QR factors for further efficiency; for details, see the "Options" section.

The above methods do not modify the input matrix `A` and may make a copy of the data in order to enforce this (whether this is necessary depends on the type of the input and the sketch method used). Potentially more efficient versions that reserve the right to overwrite `A` are available as `pqrfact!` and `pqr!`, respectively.

### Interpolative Decomposition (ID)

The ID is based on the approximation `A[:,rd] = A[:,sk]*T`, where `sk` is a set of `k` "skeleton" columns, `rd` is a set of `n - k` "redundant" columns, and `T` is a `k` by `n - k` interpolation matrix. It follows that `A[:,p] = C*V`, where `p = [sk; rd]`, `C = A[:,sk]`, and `V = [eye(k) T]`. An ID can be computed by:

```julia
V = idfact(A, args...)
```

or:

```julia
sk, rd, T = id(A, args...)
```

Here, `V` is of type `IDPackedV` and defines the `V` factor above but can also implicitly represent the entire ID via:

- `V[:sk]`: `sk` columns as type `Vector`
- `V[:rd]`: `rd` columns as type `Vector`
- `V[:p]`: `p` permutation as type `Vector`
- `V[:P]`: `p` permutation as type `ColumnPermutation`
- `V[:T]`: `T` factor as type `Matrix`

To actually produce the ID itself, use:

```julia
F = ID(A, V)
```

or:

```julia
F = ID(A, sk, rd, T)
```

which returns an `ID` factorization that can be directly compared with `A`. This factorization has access methods:

- `F[:C]`: `C` factor as type `Matrix`
- `F[:V]`: `V` factor as type `IDPackedV`

in addition to those defined for `IDPackedV`.

As with the partial QR decomposition, an ID can be computed for `A'` instead (that is, finding skeleton rows as opposed to columns) in the same way, e.g.:

```julia
V = idfact(:c, A, args...)
```

The default interface is equivalent to passing `:n` as the first argument. Moreover, modifying versions of the above are available as `idfact!` and `id!`.

### Singular Value Decomposition (SVD)

A partial SVD is a factorization `A = U*S*V'`, where `U` and `V` are `m` by `k` and `n` by `k`, respectively, both with orthonormal columns, and `S` is `k` by `k` and diagonal with nonincreasing, nonnegative, real entries. It can be computed with:

```julia
F = psvdfact(A, args...)
```

or:

```julia
U, S, V = psvd(A, args...)
```

The factorization is of type `PartialSVD` and has access methods:

- `F[:U]`: `U` factor as type `Matrix`
- `F[:S]`: `S` factor as type `Vector`
- `F[:V]`: `V` factor as type `Matrix`
- `F[:Vt]`: `V'` factor as type `Matrix`

Note that the underlying SVD routine forms `V'` as output, so `F[:Vt]` is easier to extract than `F[:V]`. Least squares solution is also supported using left-division. Furthermore, if just the singular values are required, then we can use:

```julia
S = psvdvals(A, args...)
```

### Hermitian Eigendecomposition

A partial Hermitian eigendecomposition of an `n` by `n` Hermitian matrix `A` is a factorization `A = U*S*U'`, where `U` is `n` by `k` with orthonormal columns and `S` is `k` by `k` and diagonal with nondecreasing, real entries. It is very similar to a partial Hermitian SVD and can be computed by:

```julia
F = pheigfact(A, args...)
```

or:

```julia
values, vectors = pheig(A, args...)
```

where we have followed the Julia convention of letting `values` denote the eigenvalues comprising `S` and `vectors` denote the eigenvector matrix `U`. The factorization is of type `PartialHermitianEigen` and has access methods:

- `F[:values]`: `values` as type `Vector`
- `F[:vectors]`: `vectors` as type `Matrix`

It also supports least squares solution by left-division. If only the eigenvalues are desired, use instead:

```julia
values = pheigvals(A, args...)
```

### CUR Decomposition

A CUR decomposition is a factorization `A = C*U*R`, where `C = A[:,cols]` and `R = A[rows,:]` consist of columns and rows, respectively, from `A` and `U = pinv(A[rows,cols])`. If `length(cols) = kc` and `length(rows) = kr`, then `C` is `m` by `kc`, `U` is `kc` by `kr`, and `R` is `kr` by `n`, with `k = min(kc, kr)`. The basis rows and columns can be computed with:

```julia
U = curfact(A, args...)
```

or:

```julia
rows, cols = cur(A, args...)
```

The former is of type `CURPackedU` (or `HermitianCURPackedU` if `A` is Hermitian or `SymmetricCURPackedU` if symmetric) and has access methods:

- `U[:cols]`: `cols` columns as type `Vector`
- `U[:rows]`: `rows` rows as type `Vector`
- `U[:kc]`: `kc` column rank
- `U[:kr]`: `kr` row rank

To produce the corresponding CUR decomposition, use:

```julia
F = CUR(A, U)
```

or:

```julia
F = CUR(A, rows, cols)
```

which returns a `CUR` factorization (or `HermitianCUR` if `A` is Hermitian or `SymmetricCUR` if symmetric), with access methods:

- `F[:C]`: `C` factor as type `Matrix`
- `F[:U]`: `U` factor as type `Factorization`
- `F[:R]`: `R` factor as type `Matrix`

in addition to those defined for `CURPackedU`. If `F` is of type `HermitianCUR`, then `F[:R] = F[:C]'`, while if `F` has type `SymmetricCUR`, then `F[:R] = F[:C].'`. Note that because of conditioning issues, `U` is not stored explicitly but rather in factored form, nominally as type `SVD` but practically as `PartialHermitianEigen` if `U` has type `HermitianCURPackedU` or `PartialSVD` otherwise (for convenient arithmetic operations).

Modifying versions of the above are available as `curfact!` and `cur!`.

## Sketch Methods

### Random Gaussian

### Random Subset

### Subsampled Random Fourier Transform (SRFT)

### Sparse Random Gaussian

## Other Algorithms

## Options

## Computational Complexity
