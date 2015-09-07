# LowRankApprox

[![Build Status](https://travis-ci.org/klho/LowRankApprox.jl.svg?branch=master)](https://travis-ci.org/klho/LowRankApprox.jl)

This Julia package implements fast low-rank matrix approximation algorithms for BLAS/LAPACK-compatible matrices. The focus is on performance and feature-completeness; we include both deterministic early-termination variants of standard codes as well as those based on some of the latest work in adaptive randomized matrix sketching. All user-level functions accept a number of options specifying, e.g., the rank, estimated absolute precision, and estimated relative precision of approximation.

## Usage

To illustrate the usage of this package, consider the computation of a partial QR decomposition of a matrix `A`:
```
F = LowRankApprox.pqrfact(A)
```
or, alternatively,
```
Q, R, p = LowRankApprox.pqr(A)
```
This computes a factorization using the default options: an early-terminated version of GEQP3 at approximately `1e-15` relative precision. To request only `1e-12` relative precision, use
```
F = LowRankApprox.pqrfact(A, 1e-12)
```
We can also directly control the rank of the approximation instead by typing, e.g.,
```
F = LowRankApprox.pqrfact(A, 20)
```
which computes an approximation of rank at most `20`.

Further user options can be exposed through the interface
```
F = LowRankApprox.pqrfact(A, opts)
```
For example, to compute an approximation of rank at most `20` to an estimated relative precision of at least `1e-12`, use
```
opts = LowRankApprox.LRAOptions(rank=20, rtol=1e-12)
```
In other words, this sets two termination criteria, one at reaching rank `20` and another at achieving `1e-12` relative precision, with the algorithm terminating when either one of these is reached. The most general accuracy setting includes also the estimated absolute precision, e.g.,
```
opts = LowRankApprox.LRAOptions(atol=1e-9, rank=20, rtol=1e-12)
```

We can also take advantage of matrix sketching (randomized sampling) as follows. For example, to sketch with a random Gaussian matrix, set
```
opts = LowRankApprox.LRAOptions(sketch=:randn)
```
This sketch is used to quickly compress the matrix, which is then run through the deterministic algorithm (at a potentially much lower cost) and post-processed to return the original matrix factors as requested. Sketching is a core component of this package and is completely adaptive based on the accuracy options described above.

## Status

Currently implemented algorithms include:
- sketch methods:
 - random Gaussian
 - random subset
 - subsampled random Fourier transform
 - sparse random Gaussian
- randomized range finder based on sketching
- factorizations:
 - partial QR decomposition
 - interpolative decomposition
 - partial singular value decomposition

Still to come (hopefully):
- partial eigendecomposition
- CUR decomposition