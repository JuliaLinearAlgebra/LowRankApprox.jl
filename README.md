# LowRankApprox

[![Build Status](https://travis-ci.org/klho/LowRankApprox.jl.svg?branch=master)](https://travis-ci.org/klho/LowRankApprox.jl)

This Julia package implements fast low-rank matrix approximation algorithms for BLAS/LAPACK-compatible matrices. The focus is on performance and feature-completeness; we include both deterministic early-termination variants of standard codes as well as those based on some of the latest technology in adaptive randomized matrix sketching. All user-level functions accept a number of options specifying, e.g., the rank, estimated absolute precision, and estimated relative precision of approximation.

## Usage

To illustrate the usage of this package, consider the computation of a partial QR decomposition of a matrix `A`:
```
F = LowRankApprox.pqrfact(A)
```
or, alternatively,
```
Q, R, p = LowRankApprox.pqr(A)
```
This computes a factorization using the default options: an approximation to roughly `1e-15` relative precision accelerated by an adaptive random Gaussian sketch. To request only `1e-12` relative precision, use
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
For example, to compute an approximation of rank at most `20` or to an estimated relative precision of at most `1e-12`, use
```
opts = LowRankApprox.LRAOptions(rank=20, rtol=1e-12)
```
In other words, this sets two termination criteria, one at reaching rank `20` and another at achieving `1e-12` relative precision, with the algorithm terminating when either one of these is reached. The most general accuracy setting includes also the estimated absolute precision, e.g.,
```
opts = LowRankApprox.LRAOptions(atol=1e-9, rank=20, rtol=1e-12)
```

Other matrix sketching methods can also be specified. For example, to sketch with a subsampled random Fourier transform, set
```
opts = LowRankApprox.LRAOptions(sketch=:srft)
```
Sketching is a core component of this package and is critical for performance. To disable sketching, use
```
opts = LowRankApprox.LRAOptions(sketch=:none)
```

## Status

Currently implemented algorithms include:
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
 - eigendecomposition

Still to come (hopefully):
- CUR decomposition