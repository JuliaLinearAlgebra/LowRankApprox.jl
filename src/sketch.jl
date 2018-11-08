#= src/sketch.jl

References:

  N. Halko, P.G. Martinsson, J.A. Tropp. Finding structure with randomness:
    Probabilistic algorithms for constructing approximate matrix
    decompositions. SIAM Rev. 53 (2): 217-288, 2011.

  F. Woolfe, E. Liberty, V. Rokhlin, M. Tygert. A fast randomized algorithm for
    the approximation of matrices. Appl. Comput. Harmon. Anal. 25: 335-366,
    2008.
=#

abstract type SketchMatrix end

size(A::SketchMatrix, dim::Integer...) = A.k

  function *(A::SketchMatrix, B::AbstractMatOrLinOp{T}) where T
    C = Array{T}(undef, A.k, size(B,2))
    mul!(C, A, B)
  end
  function *(A::SketchMatrix, Bc::Adjoint{T,<:AbstractMatOrLinOp{T}}) where T
    C = Array{T}(undef, A.k, size(Bc,2))
    mul!(C, A, Bc)
  end
  function *(A::AbstractMatOrLinOp{T}, B::SketchMatrix) where T
    C = Array{T}(undef, size(A,1), B.k)
    mul!(C, A, B)
  end
  function *(Ac::Adjoint{T,<:AbstractMatOrLinOp{T}}, B::SketchMatrix) where T
    C = Array{T}(undef, size(Ac,1), B.k)
    mul!(C, Ac, B)
  end

function sketch(
    side::Symbol, trans::Symbol, A::AbstractMatOrLinOp{T}, order::Integer,
    opts::LRAOptions=LRAOptions(T); args...) where T
  sketch_chkargs(side, trans, order)
  opts = copy(opts; args...)
  sketch_chkopts!(opts, A)
  if     opts.sketch == :randn  return sketch_randn(side, trans, A, order, opts)
  elseif opts.sketch == :sprn   return  sketch_sprn(side, trans, A, order, opts)
  elseif opts.sketch == :srft   return  sketch_srft(side, trans, A, order, opts)
  elseif opts.sketch == :sub    return   sketch_sub(side, trans, A, order, opts)
  end
end
sketch(side::Symbol, trans::Symbol, A, order::Integer, args...; kwargs...) =
  sketch(side, trans, LinOp(A), order, args...; kwargs...)
sketch(A, order::Integer, args...; kwargs...) =
  sketch(:left, :n, A, order, args...; kwargs...)

function sketchfact(
    side::Symbol, trans::Symbol, A::AbstractMatOrLinOp{T},
    opts::LRAOptions=LRAOptions(T); args...) where T
  sketchfact_chkargs(side, trans)
  opts = copy(opts; args...)
  sketch_chkopts!(opts, A)
  if     opts.sketch == :randn  return sketchfact_randn(side, trans, A, opts)
  elseif opts.sketch == :sprn   return  sketchfact_sprn(side, trans, A, opts)
  elseif opts.sketch == :srft   return  sketchfact_srft(side, trans, A, opts)
  elseif opts.sketch == :sub    return   sketchfact_sub(side, trans, A, opts)
  end
end
sketchfact(side::Symbol, trans::Symbol, A, args...; kwargs...) =
  sketchfact(side, trans, LinOp(A), args...; kwargs...)
sketchfact(A, args...; kwargs...) = sketchfact(:left, :n, A, args...; kwargs...)

function sketch_chkopts!(opts::LRAOptions, A)
  chkopts!(opts)
  if opts.sketch == :none
    warn("invalid sketch method; using \"randn\"")
    opts.sketch = :randn
  end
end

function sketch_chkargs(side::Symbol, trans::Symbol, order::Integer)
  sketchfact_chkargs(side, trans)
  order >= 0 || throw(ArgumentError("order"))
end
function sketchfact_chkargs(side::Symbol, trans::Symbol)
  side in (:left, :right) || throw(ArgumentError("side"))
  trans in (:n, :c) || throw(ArgumentError("trans"))
end

# RandomGaussian

mutable struct RandomGaussian <: SketchMatrix
  k::Int
end

function mul!(C, A::RandomGaussian, B::AbstractMatOrLinOp{T}) where T
  S = crandn(T, A.k, size(B,1))
  mul!(C, S, B)
end
function mul!(C, A::AbstractMatOrLinOp{T}, B::RandomGaussian) where T
  S = crandn(T, size(A,2), B.k)
  mul!(C, A, S)
end


  function mul!(C, A::RandomGaussian, Bc::Adjoint{T,<:AbstractMatOrLinOp{T}}) where T
    S = crandn(T, A.k, size(Bc,1))
    mul!(C, S, Bc)
  end


  function mul!(C, Ac::Adjoint{T,<:AbstractMatOrLinOp{T}}, B::RandomGaussian) where T
    S = crandn(T, size(Ac,2), B.k)
    mul!(C, Ac, S)
  end

## sketch interface

function sketch_randn(
    side::Symbol, trans::Symbol, A::AbstractMatOrLinOp, order::Integer,
    opts::LRAOptions)
  if side == :left
    if trans == :n  return sketch_randn_ln(A, order, opts)
    else            return sketch_randn_lc(A, order, opts)
    end
  else
    if trans == :n  return sketch_randn_rn(A, order, opts)
    else            return sketch_randn_rc(A, order, opts)
    end
  end
end


  function sketch_randn_ln(A::AbstractMatOrLinOp{T}, order::Integer, opts::LRAOptions) where T
    S = RandomGaussian(order)
    m, n = size(A)
    isherm = ishermitian(A)
    Bp = Array{T}(undef, order, n)
    if opts.sketch_randn_niter > 0
      Bq   = Array{T}(undef, order, m)
      tau  = Array{T}(undef, 1)
      work = Array{T}(undef, 1)
    end
    mul!(Bp, S, A)
    for i = 1:opts.sketch_randn_niter
      Bp, tau, work = orthrows!(Bp, tau, work, thin=false)
      mul!(Bq, Bp, A')
      if isherm
        Bp, Bq = Bq, Bp
      else
        Bq, tau, work = orthrows!(Bq, tau, work, thin=false)
        mul!(Bp, Bq, A)
      end
    end
    Bp
  end
  function sketch_randn_lc(A::AbstractMatOrLinOp{T}, order::Integer, opts::LRAOptions) where T
    S = RandomGaussian(order)
    m, n = size(A)
    isherm = ishermitian(A)
    Bp = Array{T}(undef, order, m)
    if opts.sketch_randn_niter > 0
      Bq   = Array{T}(undef, order, n)
      tau  = Array{T}(undef, 1)
      work = Array{T}(undef, 1)
    end
    mul!(Bp, S, A')
    for i = 1:opts.sketch_randn_niter
      Bp, tau, work = orthrows!(Bp, tau, work, thin=false)
      mul!(Bq, Bp, A)
      if isherm
        Bp, Bq = Bq, Bp
      else
        Bq, tau, work = orthrows!(Bq, tau, work, thin=false)
        mul!(Bp, Bq, A')
      end
    end
    Bp
  end
  function sketch_randn_rn(A::AbstractMatOrLinOp{T}, order::Integer, opts::LRAOptions) where T
    S = RandomGaussian(order)
    m, n = size(A)
    isherm = ishermitian(A)
    Bp = Array{T}(undef, m, order)
    if opts.sketch_randn_niter > 0
      Bq   = Array{T}(undef, n, order)
      tau  = Array{T}(undef, 1)
      work = Array{T}(undef, 1)
    end
    mul!(Bp, A, S)
    for i = 1:opts.sketch_randn_niter
      Bp, tau, work = orthcols!(Bp, tau, work, thin=false)
      mul!(Bq, A', Bp)
      if isherm
        Bp, Bq = Bq, Bp
      else
        Bq, tau, work = orthcols!(Bq, tau, work, thin=false)
        mul!(Bp, A, Bq)
      end
    end
    Bp
  end
  function sketch_randn_rc(A::AbstractMatOrLinOp{T}, order::Integer, opts::LRAOptions) where T
    S = RandomGaussian(order)
    m, n = size(A)
    isherm = ishermitian(A)
    Bp = Array{T}(undef, n, order)
    if opts.sketch_randn_niter > 0
      Bq   = Array{T}(undef, m, order)
      tau  = Array{T}(undef, 1)
      work = Array{T}(undef, 1)
    end
    mul!(Bp, A', S)
    for i = 1:opts.sketch_randn_niter
      Bp, tau, work = orthcols!(Bp, tau, work, thin=false)
      mul!(Bq, A, Bp)
      if isherm
        Bp, Bq = Bq, Bp
      else
        Bq, tau, work = orthcols!(Bq, tau, work, thin=false)
        mul!(Bp, A', Bq)
      end
    end
    Bp
  end


function sketchfact_randn(
    side::Symbol, trans::Symbol, A::AbstractMatOrLinOp, opts::LRAOptions)
  if opts.sketchfact_adap || opts.rank < 0
    n = opts.nb
    opts_ = copy(opts, maxdet_tol=-1.)
    while true
      order = opts.sketchfact_randn_samp(n)
      B = sketch_randn(side, trans, A, order, opts)
      p, tau, k = geqp3_adap!(B, opts_)
      k < n && return pqrback_postproc(B, p, tau, k, opts)
      n *= 2
    end
  else
    order = opts.sketchfact_randn_samp(opts.rank)
    B = sketch_randn(side, trans, A, order, opts)
    return pqrfact_backend!(B, opts)
  end
end

# RandomSubset

mutable struct RandomSubset <: SketchMatrix
  k::Int
end

function mul!(C, A::RandomSubset, B::AbstractMatrix)
  k = A.k
  m, n = size(B)
  size(C) == (k, n) || throw(DimensionMismatch)
  r = rand(1:m, k)
  for i = 1:k
    copyto!(view(C,[i],:), view(B,r[i:i],:))
  end
  C
end

function mul!(C, A::AbstractMatrix, B::RandomSubset)
  k = B.k
  m, n = size(A)
  size(C) == (m, k) || throw(DimensionMismatch)
  r = rand(1:n, k)
  for i = 1:k
    copyto!(view(C,:,[i]), view(A,:,r[i:i]))
  end
  C
end


  function mul!(C, A::RandomSubset, Bc::Adjoint{T,<:AbstractMatrix{T}}) where T
    B = parent(Bc)
    k = A.k
    m, n = size(B)
    size(C) == (k, m) || throw(DimensionMismatch)
    r = rand(1:n, k)
    for i = 1:k
      adjoint!(view(C,[i],:), view(B,:,r[i:i]))
    end
    C
  end

  function mul!(C, Ac::Adjoint{T,<:AbstractMatrix{T}}, B::RandomSubset) where T
    A = parent(Ac)
    k = B.k
    m, n = size(A)
    size(C) == (n, k) || throw(DimensionMismatch)
    r = rand(1:m, k)
    for i = 1:k
      adjoint!(view(C,:,[i]), view(A,r[i:i],:))
    end
    C
  end


## sketch interface

function sketch_sub(
    side::Symbol, trans::Symbol, A::AbstractMatrix, order::Integer,
    opts::LRAOptions)
  S = RandomSubset(order)
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
    n = opts.nb
    opts_ = copy(opts, maxdet_tol=-1.)
    while true
      order = opts.sketchfact_sub_samp(n)
      B = sketch_sub(side, trans, A, order, opts)
      p, tau, k = geqp3_adap!(B, opts_)
      k < n && return pqrback_postproc(B, p, tau, k, opts)
      n *= 2
    end
  else
    order = opts.sketchfact_sub_samp(opts.rank)
    B = sketch_sub(side, trans, A, order, opts)
    return pqrfact_backend!(B, opts)
  end
end

# SRFT

mutable struct SRFT <: SketchMatrix
  k::Int
end

function srft_rand(::Type{T}, n::Integer) where T<:Real
  x = rand(n)
  @simd for i = 1:n
    @inbounds x[i] = 2*(x[i] > 0.5) - 1
  end
  x
end
function srft_rand(::Type{T}, n::Integer) where T<:Complex
  x = crandn(T, n)
  @inbounds for i = 1:n
    x[i] /= abs(x[i])
  end
  x
end

function srft_init(::Type{T}, n::Integer, k::Integer) where T<:Real
  l = k
  while n % l > 0
    l -= 1
  end
  m = div(n, l)
  X = Array{T}(undef, l, m)
  d = srft_rand(T, n)
  idx = rand(1:n, k)
  r2rplan! = plan_r2r!(X, R2HC, 1)
  X, d, idx, r2rplan!
end
function srft_init(::Type{T}, n::Integer, k::Integer) where T<:Complex
  l = k
  while n % l > 0
    l -= 1
  end
  m = div(n, l)
  X = Array{T}(undef, l, m)
  d = srft_rand(T, n)
  idx = rand(1:n, k)
  fftplan! = plan_fft!(X, 1)
  X, d, idx, fftplan!
end

function srft_reshape!(X::AbstractMatrix, d::AbstractVector, x::AbstractVecOrMat)
  l, m = size(X)
  i = 0
  @inbounds for j = 1:l, k = 1:m
    i += 1
    X[j,k] = d[i]*x[i]
  end
end
function srft_reshape_conj!(
    X::AbstractMatrix, d::AbstractVector, x::AbstractVecOrMat)
  l, m = size(X)
  i = 0
  @inbounds for j = 1:l, k = 1:m
    i += 1
    X[j,k] = d[i]*conj(x[i])
  end
end

function srft_apply!(
    y::AbstractVecOrMat{T}, X::AbstractMatrix{T}, idx::AbstractVector,
    r2rplan!::r2rFFTWPlan) where T<:Real
  l, m = size(X)
  n = l*m
  k = length(idx)
  mul!(X, r2rplan!, X)
  wn = exp(-2im*pi/n)
  wm = exp(-2im*pi/m)
  nnyq = div(n, 2)
  cnyq = div(l, 2)
  i = 1
  @inbounds while i <= k
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
    s = one(T)

    # compute only one entry if purely real or no more space
    if in == 0 || in == nnyq || i == k
      for j = 1:m
        a = X[ia+1,j]
        b = ib == 0 || ib == ia ? zero(T) : X[ib+1,j]
        b = cswap ? -b : b
        y[i] += real(s*(a + b*im))
        s *= w
      end

    # else compute one entry each for real/imag parts
    else
      y[i+1] = 0
      for j = 1:m
        a = X[ia+1,j]
        b = ib == 0 || ib == ia ? zero(T) : X[ib+1,j]
        b = cswap ? -b : b
        z = s*(a + b*im)
        y[i  ] += real(z)
        y[i+1] += imag(z)
        s *= w
      end
      i += 1
    end
    i += 1
  end
end

function srft_apply!(
    y::AbstractVecOrMat{T}, X::AbstractMatrix, idx::AbstractVector,
    fftplan!::FFTWPlan) where T<:Complex
  l, m = size(X)
  n = l*m
  k = length(idx)
  mul!(X, fftplan!, X)
  wn = exp(-2im*pi/n)
  wm = exp(-2im*pi/m)
  @inbounds for i = 1:k
    row = fld(idx[i] - 1, l) + 1
    col = rem(idx[i] - 1, l) + 1
    w = wm^(row - 1)*wn^(col - 1)
    y[i] = 0
    s = one(T)
    for j = 1:m
      y[i] += s*X[col,j]
      s *= w
    end
  end
end

function mul!(C, A::SRFT, B::AbstractMatrix{T}) where T
  m, n = size(B)
  k = A.k
  size(C) == (k, n) || throw(DimensionMismatch)
  X, d, idx, fftplan! = srft_init(T, m, k)
  for i = 1:n
    srft_reshape!(X, d, view(B,:,i))
    srft_apply!(view(C,:,i), X, idx, fftplan!)
  end
  C
end

function mul!(C, A::AbstractMatrix{T}, B::SRFT) where T
  m, n = size(A)
  k = B.k
  size(C) == (m, k) || throw(DimensionMismatch)
  X, d, idx, fftplan! = srft_init(T, n, k)
  for i = 1:m
    srft_reshape!(X, d, view(A,i,:))
    srft_apply!(view(C,i,:), X, idx, fftplan!)
  end
  C
end


  function mul!(C, A::SRFT, Bc::Adjoint{T,<:AbstractMatrix{T}}) where T
    B = parent(Bc)
    m, n = size(B)
    k = A.k
    size(C) == (k, m) || throw(DimensionMismatch)
    X, d, idx, fftplan! = srft_init(T, n, k)
    for i = 1:m
      srft_reshape_conj!(X, d, view(B,i,:))
      srft_apply!(view(C,:,i), X, idx, fftplan!)
    end
    C
  end
  function mul!(C, Ac::Adjoint{T,<:AbstractMatrix{T}}, B::SRFT) where T
    A = parent(Ac)
    m, n = size(A)
    k = B.k
    size(C) == (n, k) || throw(DimensionMismatch)
    X, d, idx, fftplan! = srft_init(T, m, k)
    for i = 1:n
      srft_reshape_conj!(X, d, view(A,:,i))
      srft_apply!(view(C,i,:), X, idx, fftplan!)
    end
    C
  end





## sketch interface

function sketch_srft(
    side::Symbol, trans::Symbol, A::AbstractMatrix, order::Integer,
    opts::LRAOptions)
  S = SRFT(order)
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
    side::Symbol, trans::Symbol, A::AbstractMatrix, opts::LRAOptions)
  if opts.sketchfact_adap || opts.rank < 0
    n = opts.nb
    opts_ = copy(opts, maxdet_tol=-1.)
    while true
      order = opts.sketchfact_srft_samp(n)
      B = sketch_srft(side, trans, A, order, opts)
      p, tau, k = geqp3_adap!(B, opts_)
      k < n && return pqrback_postproc(B, p, tau, k, opts)
      n *= 2
    end
  else
    order = opts.sketchfact_srft_samp(opts.rank)
    B = sketch_srft(side, trans, A, order, opts)
    return pqrfact_backend!(B, opts)
  end
end

# SparseRandomGaussian

mutable struct SparseRandomGaussian <: SketchMatrix
  k::Int
end
const SparseRandGauss = SparseRandomGaussian

function mul!(C, A::SparseRandGauss, B::AbstractMatrix{T}) where T
  k = A.k
  m, n = size(B)
  size(C) == (k, n) || throw(DimensionMismatch)
  r = randperm(m)
  idx = 0
  @inbounds for i = 1:k
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


function mul!(C, A::AbstractMatrix{T}, B::SparseRandGauss) where T
  k = B.k
  m, n = size(A)
  size(C) == (m, k) || throw(DimensionMismatch)
  r = randperm(n)
  idx = 0
  @inbounds for j = 1:k
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



  function mul!(C, A::SparseRandGauss, Bc::Adjoint{T,<:AbstractMatrix{T}}) where T
    B = parent(Bc)
    k = A.k
    m, n = size(B)
    size(C) == (k, m) || throw(DimensionMismatch)
    r = randperm(n)
    idx = 0
    @inbounds for i = 1:k
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
  function mul!(C, Ac::Adjoint{T,<:AbstractMatrix{T}}, B::SparseRandGauss) where T
    A = parent(Ac)
    k = B.k
    m, n = size(A)
    size(C) == (n, k) || throw(DimensionMismatch)
    r = randperm(m)
    idx = 0
    @inbounds for j = 1:k
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
    side::Symbol, trans::Symbol, A::AbstractMatrix, order::Integer,
    opts::LRAOptions)
  S = SparseRandGauss(order)
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
    n = opts.nb
    opts_ = copy(opts, maxdet_tol=-1.)
    while true
      B = sketch_sprn(side, trans, A, n, opts)
      p, tau, k = geqp3_adap!(B, opts_)
      k < n && return pqrback_postproc(B, p, tau, k, opts)
      n *= 2
    end
  else
    order = opts.rank
    B = sketch_sprn(side, trans, A, order, opts)
    return pqrfact_backend!(B, opts)
  end
end
