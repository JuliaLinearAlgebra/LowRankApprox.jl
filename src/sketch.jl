#= src/sketch.jl

References:

  N. Halko, P.G. Martinsson, J.A. Tropp. Finding structure with randomness:
    Probabilistic algorithms for constructing approximate matrix
    decompositions. SIAM Rev. 53 (2): 217-288, 2011.

  F. Woolfe, E. Liberty, V. Rokhlin, M. Tygert. A fast randomized algorithm for
    the approximation of matrices. Appl. Comput. Harmon. Anal. 25: 335-366,
    2008.
=#

abstract SketchMatrix

size(A::SketchMatrix, dim::Integer...) = A.k

for (f, f!, i) in ((:*,        :A_mul_B!,  2),
                   (:A_mul_Bc, :A_mul_Bc!, 1))
  @eval begin
    function $f{T}(A::SketchMatrix, B::AbstractMatOrLinOp{T})
      C = Array(T, A.k, size(B,$i))
      $f!(C, A, B)
    end
  end
end

for (f, f!, i) in ((:*,        :A_mul_B!,  1),
                   (:Ac_mul_B, :Ac_mul_B!, 2))
  @eval begin
    function $f{T}(A::AbstractMatOrLinOp{T}, B::SketchMatrix)
      C = Array(T, size(A,$i), B.k)
      $f!(C, A, B)
    end
  end
end

function sketch(
    side::Symbol, trans::Symbol, A::AbstractMatOrLinOp, rank::Integer,
    opts::LRAOptions)
  opts = sketch_chkopts(A, opts)
  sketch_chkargs(side, trans, rank)
  if     opts.sketch == :randn  return sketch_randn(side, trans, A, rank, opts)
  elseif opts.sketch == :sprn   return  sketch_sprn(side, trans, A, rank, opts)
  elseif opts.sketch == :srft   return  sketch_srft(side, trans, A, rank, opts)
  elseif opts.sketch == :sub    return   sketch_sub(side, trans, A, rank, opts)
  end
end
function sketch(
    side::Symbol, trans::Symbol, A::AbstractMatOrLinOp, rank::Integer)
  opts = LRAOptions(sketch=:randn)
  sketch(side, trans, A, rank, opts)
end
sketch(side::Symbol, trans::Symbol, A, rank::Integer, args...) =
  sketch(side, trans, LinOp(A), rank, args...)
sketch(A, rank::Integer, args...) = sketch(:left, :n, A, rank, args...)

function sketchfact(
    side::Symbol, trans::Symbol, A::AbstractMatOrLinOp, opts::LRAOptions)
  opts = sketch_chkopts(A, opts)
  sketchfact_chkargs(side, trans)
  if     opts.sketch == :randn  return sketchfact_randn(side, trans, A, opts)
  elseif opts.sketch == :sprn   return  sketchfact_sprn(side, trans, A, opts)
  elseif opts.sketch == :srft   return  sketchfact_srft(side, trans, A, opts)
  elseif opts.sketch == :sub    return   sketchfact_sub(side, trans, A, opts)
  end
end
function sketchfact(
    side::Symbol, trans::Symbol, A::AbstractMatOrLinOp, rank_or_rtol::Real)
  opts = (rank_or_rtol < 1 ? LRAOptions(rtol=rank_or_rtol)
                           : LRAOptions(rank=rank_or_rtol))
  sketchfact(side, trans, A, opts)
end
sketchfact{T}(side::Symbol, trans::Symbol, A::AbstractMatOrLinOp{T}) =
  sketchfact(side, trans, A, default_rtol(T))
sketchfact(side::Symbol, trans::Symbol, A, args...) =
  sketchfact(side, trans, LinOp(A), args...)
sketchfact(A, args...) = sketchfact(:left, :n, A, args...)

function sketch_chkopts(A, opts::LRAOptions)
  chkopts(opts)
  if opts.sketch == :none || typeof(A) <: AbstractLinOp && opts.sketch != :randn
    warn("invalid sketch method; using \"randn\"")
    opts = copy(opts, sketch=:randn)
  end
  opts
end

function sketch_chkargs(side::Symbol, trans::Symbol, rank::Integer)
  sketchfact_chkargs(side, trans)
  rank >= 0 || throw(ArgumentError("rank"))
end
function sketchfact_chkargs(side::Symbol, trans::Symbol)
  side in (:left, :right) || throw(ArgumentError("side"))
  trans in (:n, :c) || throw(ArgumentError("trans"))
end

# RandomGaussian

type RandomGaussian <: SketchMatrix
  k::Int
end

full{T}(::Type{T}, side::Symbol, A::RandomGaussian, n::Integer) =
  side == :left ? crandn(T, A.k, n) : crandn(T, n, A.k)

A_mul_B!{T}(C, A::RandomGaussian, B::AbstractMatOrLinOp{T}) =
  (S = full(T, :left, A, size(B,1)); A_mul_B!(C, S, B))
A_mul_Bc!{T}(C, A::RandomGaussian, B::AbstractMatOrLinOp{T}) =
  (S = full(T, :left, A, size(B,2)); A_mul_Bc!(C, S, B))

A_mul_B!{T}(C, A::AbstractMatOrLinOp{T}, B::RandomGaussian) =
  (S = full(T, :right, B, size(A,2)); A_mul_B!(C, A, S))
Ac_mul_B!{T}(C, A::AbstractMatOrLinOp{T}, B::RandomGaussian) =
  (S = full(T, :right, B, size(A,1)); Ac_mul_B!(C, A, S))

## sketch interface

function sketch_randn(
    side::Symbol, trans::Symbol, A::AbstractMatOrLinOp, k::Integer,
    opts::LRAOptions)
  if side == :left
    if trans == :n  return sketch_randn_ln(A, k, opts)
    else            return sketch_randn_lc(A, k, opts)
    end
  else
    if trans == :n  return sketch_randn_rn(A, k, opts)
    else            return sketch_randn_rc(A, k, opts)
    end
  end
end

for (trans, p, q, g, h) in ((:n, :n, :m, :A_mul_B!,  :A_mul_Bc!),
                            (:c, :m, :n, :A_mul_Bc!, :A_mul_B! ))
  f = symbol("sketch_randn_l", trans)
  @eval begin
    function $f{T}(A::AbstractMatOrLinOp{T}, k::Integer, opts::LRAOptions)
      S = RandomGaussian(k)
      m, n = size(A)
      isherm = ishermitian(A)
      Bp = Array(T, k, $p)
      if opts.sketch_randn_niter > 0
        Bq = Array(T, k, $q)
      end
      $g(Bp, S, A)
      for i = 1:opts.sketch_randn_niter
        $h(Bq, orthrows!(Bp, thin=false), A)
        if isherm  Bp, Bq = Bq, Bp
        else       $g(Bp, orthrows!(Bq, thin=false), A)
        end
      end
      Bp
    end
  end
end

for (trans, p, q, g, h) in ((:n, :m, :n, :A_mul_B!,  :Ac_mul_B!),
                            (:c, :n, :m, :Ac_mul_B!, :A_mul_B! ))
  f = symbol("sketch_randn_r", trans)
  @eval begin
    function $f{T}(A::AbstractMatOrLinOp{T}, k::Integer, opts::LRAOptions)
      S = RandomGaussian(k)
      m, n = size(A)
      isherm = ishermitian(A)
      Bp = Array(T, $p, k)
      if opts.sketch_randn_niter > 0
        Bq = Array(T, $q, k)
      end
      $g(Bp, A, S)
      for i = 1:opts.sketch_randn_niter
        $h(Bq, A, orthcols!(Bp, thin=false))
        if isherm  Bp, Bq = Bq, Bp
        else       $g(Bp, A, orthcols!(Bq, thin=false))
        end
      end
      Bp
    end
  end
end

function sketchfact_randn(
    side::Symbol, trans::Symbol, A::AbstractMatOrLinOp, opts::LRAOptions)
  if opts.sketchfact_adap || opts.rank < 0
    k = opts.nb
    while true
      B = sketch_randn(side, trans, A, k+opts.sketchfact_randn_samp, opts)
      F = pqrfact_lapack!(B, opts)
      F[:k] < k && return F
      k *= 2
    end
  else
    k = opts.rank + opts.sketchfact_randn_samp
    B = sketch_randn(side, trans, A, k, opts)
    return pqrfact_lapack!(B, opts)
  end
end

# RandomSubset

type RandomSubset <: SketchMatrix
  k::Int
end

function A_mul_B!(C, A::RandomSubset, B::AbstractMatrix)
  k = A.k
  m, n = size(B)
  size(C) == (k, n) || throw(DimensionMismatch)
  r = rand(1:m, k)
  for i = 1:k
    copy!(sub(C,i,:), sub(B,r[i],:))
  end
  C
end
function A_mul_Bc!(C, A::RandomSubset, B::AbstractMatrix)
  k = A.k
  m, n = size(B)
  size(C) == (k, m) || throw(DimensionMismatch)
  r = rand(1:n, k)
  for i = 1:k
    ctranspose!(sub(C,i,:), sub(B,:,r[i]))
  end
  C
end

function A_mul_B!(C, A::AbstractMatrix, B::RandomSubset)
  k = B.k
  m, n = size(A)
  size(C) == (m, k) || throw(DimensionMismatch)
  r = rand(1:n, k)
  for i = 1:k
    copy!(sub(C,:,i), sub(A,:,r[i]))
  end
  C
end
function Ac_mul_B!(C, A::AbstractMatrix, B::RandomSubset)
  k = B.k
  m, n = size(A)
  size(C) == (n, k) || throw(DimensionMismatch)
  r = rand(1:m, k)
  for i = 1:k
    ctranspose!(sub(C,:,i), sub(A,r[i],:))
  end
  C
end

## sketch interface

function sketch_sub(
    side::Symbol, trans::Symbol, A::AbstractMatrix, k::Integer,
    opts::LRAOptions)
  S = RandomSubset(k)
  if side == :left
    if trans == :n  return S*A
    else            return S*A'
    end
  else
    if trans == :n  return A *S
    else            return A'*S
    end
  end
end

function sketchfact_sub(
    side::Symbol, trans::Symbol, A::AbstractMatrix, opts::LRAOptions)
  if opts.sketchfact_adap || opts.rank < 0
    k = opts.nb
    while true
      B = sketch_sub(side, trans, A, k*opts.sketchfact_sub_samp, opts)
      F = pqrfact_lapack!(B, opts)
      F[:k] < k && return F
      k *= 2
    end
  else
    k = opts.rank*opts.sketchfact_sub_samp
    B = sketch_sub(side, trans, A, k, opts)
    return pqrfact_lapack!(B, opts)
  end
end

# SRFT

type SRFT <: SketchMatrix
  k::Int
end

srft_rand{T<:Real}(::Type{T}, n::Integer) = 2*bitrand(n) - 1
srft_rand{T<:Complex}(::Type{T}, n::Integer) = exp(2im*pi*rand(n))

function srft_init{T<:Real}(::Type{T}, n::Integer, k::Integer)
  l = k
  while n % l > 0
    l -= 1
  end
  m = div(n, l)
  X = Array(T, l, m)
  d = srft_rand(T, n)
  idx = rand(1:n, k)
  r2rplan! = FFTW.plan_r2r!(X, FFTW.R2HC, 1)
  X, d, idx, r2rplan!
end
function srft_init{T<:Complex}(::Type{T}, n::Integer, k::Integer)
  l = k
  while n % l > 0
    l -= 1
  end
  m = div(n, l)
  X = Array(T, l, m)
  d = srft_rand(T, n)
  idx = rand(1:n, k)
  fftplan! = plan_fft!(X, 1)
  X, d, idx, fftplan!
end

function srft_reshape!(X::StridedMatrix, d::AbstractVector, x::AbstractVecOrMat)
  l, m = size(X)
  n = l*m
  i = 0
  for j = 1:l, k = 1:m
    i += 1
    X[j,k] = d[i]*x[i]
  end
end
function srft_reshape_conj!(
    X::StridedMatrix, d::AbstractVector, x::AbstractVecOrMat)
  l, m = size(X)
  n = l*m
  i = 0
  for j = 1:l, k = 1:m
    i += 1
    X[j,k] = d[i]*conj(x[i])
  end
end

function srft_apply!{T<:Real}(
    y::StridedVecOrMat{T}, X::StridedMatrix{T}, idx::AbstractVector,
    r2rplan!::FFTW.r2rFFTWPlan)
  l, m = size(X)
  n = l*m
  k = length(idx)
  A_mul_B!(X, r2rplan!, X)
  wn = exp(-2im*pi/n)
  wm = exp(-2im*pi/m)
  nnyq = div(n, 2)
  cnyq = div(l, 2)
  i = 1
  while i <= k
    idx_ = idx[i] - 1
    row = fld(idx_, l) + 1
    col = rem(idx_, l) + 1
    w = wm^(row - 1)*wn^(col - 1)

    # find indices of real/imag parts
    col_ = col - 1
    cswap = col_ > cnyq
    ia = cswap ? l - col_ : col_
    ib = col_ > 0 ? l - ia : 0

    # initialze next entry to fill
    y[i] = 0

    # compute only one entry if purely real or no more space
    if in == 0 || in == nnyq || i == k
      for j = 1:m
        a = X[ia+1,j]
        b = ib == 0 || ib == ia ? zero(T) : X[ib+1,j]
        b = cswap ? -b : b
        y[i] += real(w^(j - 1)*(a + b*im))
      end

    # else compute one entry each for real/imag parts
    else
      y[i+1] = 0
      for j = 1:m
        a = X[ia+1,j]
        b = ib == 0 || ib == ia ? zero(T) : X[ib+1,j]
        b = cswap ? -b : b
        z = w^(j - 1)*(a + b*im)
        y[i  ] += real(z)
        y[i+1] += imag(z)
      end
      i += 1
    end
    i += 1
  end
end

function srft_apply!{T<:Complex}(
    y::StridedVecOrMat{T}, X::StridedMatrix, idx::AbstractVector,
    fftplan!::FFTW.FFTWPlan)
  l, m = size(X)
  n = l*m
  k = length(idx)
  A_mul_B!(X, fftplan!, X)
  wn = exp(-2im*pi/n)
  wm = exp(-2im*pi/m)
  for i = 1:k
    row = fld(idx[i] - 1, l) + 1
    col = rem(idx[i] - 1, l) + 1
    w = wm^(row - 1)*wn^(col - 1)
    y[i] = 0
    for j = 1:m
      y[i] += w^(j - 1)*X[col,j]
    end
  end
end

function A_mul_B!{T}(C, A::SRFT, B::StridedMatrix{T})
  m, n = size(B)
  k = A.k
  size(C) == (k, n) || throw(DimensionMismatch)
  X, d, idx, fftplan! = srft_init(T, m, k)
  for i = 1:n
    srft_reshape!(X, d, sub(B,:,i))
    srft_apply!(sub(C,:,i), X, idx, fftplan!)
  end
  C
end
function A_mul_Bc!{T}(C, A::SRFT, B::StridedMatrix{T})
  m, n = size(B)
  k = A.k
  size(C) == (k, m) || throw(DimensionMismatch)
  X, d, idx, fftplan! = srft_init(T, n, k)
  for i = 1:m
    srft_reshape_conj!(X, d, sub(B,i,:))
    srft_apply!(sub(C,:,i), X, idx, fftplan!)
  end
  C
end

function A_mul_B!{T}(C, A::StridedMatrix{T}, B::SRFT)
  m, n = size(A)
  k = B.k
  size(C) == (m, k) || throw(DimensionMismatch)
  X, d, idx, fftplan! = srft_init(T, n, k)
  for i = 1:m
    srft_reshape!(X, d, sub(A,i,:))
    srft_apply!(sub(C,i,:), X, idx, fftplan!)
  end
  C
end
function Ac_mul_B!{T}(C, A::StridedMatrix{T}, B::SRFT)
  m, n = size(A)
  k = B.k
  size(C) == (n, k) || throw(DimensionMismatch)
  X, d, idx, fftplan! = srft_init(T, m, k)
  for i = 1:n
    srft_reshape_conj!(X, d, sub(A,:,i))
    srft_apply!(sub(C,i,:), X, idx, fftplan!)
  end
  C
end

## sketch interface

function sketch_srft(
    side::Symbol, trans::Symbol, A::StridedMatrix, k::Integer, opts::LRAOptions)
  S = SRFT(k)
  if side == :left
    if trans == :n  return S*A
    else            return S*A'
    end
  else
    if trans == :n  return A *S
    else            return A'*S
    end
  end
end

function sketchfact_srft(
    side::Symbol, trans::Symbol, A::StridedMatrix, opts::LRAOptions)
  if opts.sketchfact_adap || opts.rank < 0
    k = opts.nb
    while true
      B = sketch_srft(side, trans, A, k+opts.sketchfact_srft_samp, opts)
      F = pqrfact_lapack!(B, opts)
      F[:k] < k && return F
      k *= 2
    end
  else
    k = opts.rank + opts.sketchfact_srft_samp
    B = sketch_srft(side, trans, A, k, opts)
    return pqrfact_lapack!(B, opts)
  end
end

# SparseRandomGaussian

type SparseRandomGaussian <: SketchMatrix
  k::Int
end
typealias SparseRandGauss SparseRandomGaussian

function A_mul_B!{T}(C, A::SparseRandGauss, B::AbstractMatrix{T})
  k = A.k
  m, n = size(B)
  size(C) == (k, n) || throw(DimensionMismatch)
  r = randperm(m)
  idx = 0
  for i = 1:k
    p = fld(m - i, k) + 1
    s = crandn(T, p)
    for j = 1:n
      C[i,j] = 0
      for l = 1:p
        C[i,j] += s[l]*B[r[idx+l],j]
      end
    end
    idx += p
  end
  C
end
function A_mul_Bc!{T}(C, A::SparseRandGauss, B::AbstractMatrix{T})
  k = A.k
  m, n = size(B)
  size(C) == (k, m) || throw(DimensionMismatch)
  r = randperm(n)
  idx = 0
  for i = 1:k
    p = fld(n - i, k) + 1
    s = crandn(T, p)
    for j = 1:m
      C[i,j] = 0
      for l = 1:p
        C[i,j] += s[l]*conj(B[j,r[idx+l]])
      end
    end
    idx += p
  end
  C
end

function A_mul_B!{T}(C, A::AbstractMatrix{T}, B::SparseRandGauss)
  k = B.k
  m, n = size(A)
  size(C) == (m, k) || throw(DimensionMismatch)
  r = randperm(n)
  idx = 0
  for j = 1:k
    p = fld(n - j, k) + 1
    s = crandn(T, p)
    for i = 1:m
      C[i,j] = 0
      for l = 1:p
        C[i,j] += A[i,r[idx+l]]*s[l]
      end
    end
    idx += p
  end
  C
end
function Ac_mul_B!{T}(C, A::AbstractMatrix{T}, B::SparseRandGauss)
  k = B.k
  m, n = size(A)
  size(C) == (n, k) || throw(DimensionMismatch)
  r = randperm(m)
  idx = 0
  for j = 1:k
    p = fld(m - j, k) + 1
    s = crandn(T, p)
    for i = 1:n
      C[i,j] = 0
      for l = 1:p
        C[i,j] += conj(A[r[idx+l],i])*s[l]
      end
    end
    idx += p
  end
  C
end

## sketch interface

function sketch_sprn(
    side::Symbol, trans::Symbol, A::AbstractMatrix, k::Integer,
    opts::LRAOptions)
  S = SparseRandGauss(k)
  if side == :left
    if trans == :n  return S*A
    else            return S*A'
    end
  else
    if trans == :n  return A *S
    else            return A'*S
    end
  end
end

function sketchfact_sprn(
    side::Symbol, trans::Symbol, A::AbstractMatrix, opts::LRAOptions)
  if opts.sketchfact_adap || opts.rank < 0
    k = opts.nb
    while true
      B = sketch_sprn(side, trans, A, k, opts)
      F = pqrfact_lapack!(B, opts)
      F[:k] < k && return F
      k *= 2
    end
  else
    k = opts.rank
    B = sketch_sprn(side, trans, A, k, opts)
    return pqrfact_lapack!(B, opts)
  end
end