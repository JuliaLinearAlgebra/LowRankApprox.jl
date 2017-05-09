#= src/pqr.jl

References:

  P. Businger, G.H. Golub. Linear least squares solutions by Householder
    transformations. Numer. Math. 7: 269-276, 1965.

  M. Gu, S.C. Eisenstat. Efficient algorithms for computing a strong
    rank-revealing QR factorization. SIAM J. Sci. Comput. 17 (4): 848-869, 1996.
=#

# PartialQRFactors

type PartialQRFactors
  Q::Nullable{Matrix}
  R::Nullable{Matrix}
  p::Vector{Int}
  k::Int
  T::Nullable{Matrix}
end
@compat const PQRFactors = PartialQRFactors

function getindex(A::PQRFactors, d::Symbol)
  if     d == :P  return ColumnPermutation(A.p)
  elseif d == :Q  return get(A.Q)
  elseif d == :R  return UpperTrapezoidal(get(A.R))
  elseif d == :T  return get(A.T)
  elseif d == :k  return A.k
  elseif d == :p  return A.p
  else            throw(KeyError(d))
  end
end

# PartialQR

type PartialQR{T} <: Factorization{T}
  Q::Matrix{T}
  R::Matrix{T}
  p::Vector{Int}
end

conj!(A::PartialQR) = PartialQR(conj!(A.Q), conj!(A.R), A.p)
conj(A::PartialQR) = PartialQR(conj(A.Q), conj(A.R), A.p)

convert{T}(::Type{PartialQR{T}}, A::PartialQR) =
  PartialQR(convert(Array{T}, A.Q), convert(Array{T}, A.R), A.p)
convert{T}(::Factorization{T}, A::PartialQR) = convert(PartialQR{T}, A)
convert(::Type{Array}, A::PartialQR) = full(A)
convert{T}(::Type{Array{T}}, A::PartialQR) = convert(Array{T}, full(A))

copy(A::PartialQR) = PartialQR(copy(A.Q), copy(A.R), copy(A.p))

full(A::PartialQR) = A_mul_Bc!(A[:Q]*A[:R], A[:P])

function getindex(A::PartialQR, d::Symbol)
  if     d == :P  return ColumnPermutation(A.p)
  elseif d == :Q  return A.Q
  elseif d == :R  return UpperTrapezoidal(A.R)
  elseif d == :k  return size(A.Q, 2)
  elseif d == :p  return A.p
  else            throw(KeyError(d))
  end
end

ishermitian(::PartialQR) = false
issym(::PartialQR) = false

isreal{T}(::PartialQR{T}) = T <: Real

ndims(::PartialQR) = 2

size(A::PartialQR) = (size(A.Q,1), size(A.R,2))
size(A::PartialQR, dim::Integer) =
  dim == 1 ? size(A.Q,1) : (dim == 2 ? size(A.R,2) : 1)

# BLAS/LAPACK multiplication/division routines

## left-multiplication

function A_mul_B!!{T}(
    C::StridedVecOrMat{T}, A::PartialQR{T}, B::StridedVecOrMat{T})
  Ac_mul_B!(A[:P], B)
  A_mul_B!(C, A[:Q], A[:R]*B)
end  # overwrites B
A_mul_B!{T}(C::StridedVecOrMat{T}, A::PartialQR{T}, B::StridedVecOrMat{T}) =
  A_mul_B!!(C, A, copy(B))

for (f!, g) in ((:A_mul_Bc!, :Ac_mul_Bc), (:A_mul_Bt!, :At_mul_Bt))
  @eval begin
    function $f!{T}(C::StridedMatrix{T}, A::PartialQR{T}, B::StridedMatrix{T})
      tmp = $g(A[:P], B)
      A_mul_B!(C, A[:Q], A[:R]*tmp)
    end
  end
end

for f in (:Ac_mul_B, :At_mul_B)
  f! = Symbol(f, "!")
  @eval begin
    function $f!{T}(
        C::StridedVecOrMat{T}, A::PartialQR{T}, B::StridedVecOrMat{T})
      tmp = $f(A[:Q], B)
      $f!(C, A[:R], tmp)
      A_mul_B!(A[:P], C)
    end
  end
end

for (f, g!) in ((:Ac_mul_Bc, :Ac_mul_B!), (:At_mul_Bt, :At_mul_B!))
  f! = Symbol(f, "!")
  @eval begin
    function $f!{T}(C::StridedMatrix{T}, A::PartialQR{T}, B::StridedMatrix{T})
      tmp = $f(A[:Q], B)
      $g!(C, A[:R], tmp)
      A_mul_B!(A[:P], C)
    end
  end
end

## right-multiplication

function A_mul_B!{T}(C::StridedMatrix{T}, A::StridedMatrix{T}, B::PartialQR{T})
  A_mul_B!(C, A*B[:Q], B[:R])
  A_mul_Bc!(C, B[:P])
end

for f in (:A_mul_Bc, :A_mul_Bt)
  f!  = Symbol(f, "!")
  f!! = Symbol(f, "!!")
  @eval begin
    function $f!!{T}(C::StridedMatrix{T}, A::StridedMatrix{T}, B::PartialQR{T})
      A_mul_B!(A, B[:P])
      tmp = $f(A, B[:R])
      $f!(C, tmp, B[:Q])
    end  # overwrites A
    $f!{T}(C::StridedMatrix{T}, A::StridedMatrix{T}, B::PartialQR{T}) =
      $f!!(C, copy(A), B)
  end
end

for f in (:Ac_mul_B, :At_mul_B)
  f! = Symbol(f, "!")
  @eval begin
    function $f!{T}(C::StridedMatrix{T}, A::StridedMatrix{T}, B::PartialQR{T})
      tmp = $f(A, B[:Q])
      A_mul_B!(C, tmp, B[:R])
      A_mul_Bc!(C, B[:P])
    end
  end
end

for (f!, g, h) in ((:Ac_mul_Bc!, :Ac_mul_B, :A_mul_Bc),
                   (:At_mul_Bt!, :At_mul_B, :A_mul_Bt))
  h! = Symbol(h, "!")
  @eval begin
    function $f!{T}(C::StridedMatrix{T}, A::StridedMatrix{T}, B::PartialQR{T})
      tmp = $g(A, B[:P])
      tmp = $h(tmp, B[:R])
      $h!(C, tmp, B[:Q])
    end
  end
end

## left-division (pseudoinverse left-multiplication)
function A_ldiv_B!{T}(
    C::StridedVecOrMat{T}, A::PartialQR{T}, B::StridedVecOrMat{T})
  tmp = (A[:R]*A.R')\(A[:Q]'*B)
  Ac_mul_B!(C, A[:R], tmp);
  A_mul_B!(A[:P], C)
end

# standard operations

## left-multiplication

for (f, f!, i) in ((:*,        :A_mul_B!,  1),
                   (:Ac_mul_B, :Ac_mul_B!, 2),
                   (:At_mul_B, :At_mul_B!, 2))
  @eval begin
    function $f{TA,TB}(A::PartialQR{TA}, B::StridedVector{TB})
      T = promote_type(TA, TB)
      AT = convert(PartialQR{T}, A)
      BT = (T == TB ? B : convert(Array{T}, B))
      CT = Array{T}(size(A,$i))
      $f!(CT, AT, BT)
    end
  end
end

for (f, f!, i, j) in ((:*,         :A_mul_B!,   1, 2),
                      (:A_mul_Bc,  :A_mul_Bc!,  1, 1),
                      (:A_mul_Bt,  :A_mul_Bt!,  1, 1),
                      (:Ac_mul_B,  :Ac_mul_B!,  2, 2),
                      (:Ac_mul_Bc, :Ac_mul_Bc!, 2, 1),
                      (:At_mul_B,  :At_mul_B!,  2, 2),
                      (:At_mul_Bt, :At_mul_Bt!, 2, 1))
  @eval begin
    function $f{TA,TB}(A::PartialQR{TA}, B::StridedMatrix{TB})
      T = promote_type(TA, TB)
      AT = convert(PartialQR{T}, A)
      BT = (T == TB ? B : convert(Array{T}, B))
      CT = Array{T}(size(A,$i), size(B,$j))
      $f!(CT, AT, BT)
    end
  end
end

## right-multiplication
for (f, f!, i, j) in ((:*,         :A_mul_B!,   1, 2),
                      (:A_mul_Bc,  :A_mul_Bc!,  1, 1),
                      (:A_mul_Bt,  :A_mul_Bt!,  1, 1),
                      (:Ac_mul_B,  :Ac_mul_B!,  2, 2),
                      (:Ac_mul_Bc, :Ac_mul_Bc!, 2, 1),
                      (:At_mul_B,  :At_mul_B!,  2, 2),
                      (:At_mul_Bt, :At_mul_Bt!, 2, 1))
  @eval begin
    function $f{TA,TB}(A::StridedMatrix{TA}, B::PartialQR{TB})
      T = promote_type(TA, TB)
      AT = (T == TA ? A : convert(Array{T}, A))
      BT = convert(PartialQR{T}, B)
      CT = Array{T}(size(A,$i), size(B,$j))
      $f!(CT, AT, BT)
    end
  end
end

## left-division
function \{TA,TB}(A::PartialQR{TA}, B::StridedVector{TB})
  T = promote_type(TA, TB)
  AT = convert(PartialQR{T}, A)
  BT = (T == TB ? B : convert(Array{T}, B))
  CT = Array{T}(size(A,2))
  A_ldiv_B!(CT, AT, BT)
end
function \{TA,TB}(A::PartialQR{TA}, B::StridedMatrix{TB})
  T = promote_type(TA, TB)
  AT = convert(PartialQR{T}, A)
  BT = (T == TB ? B : convert(Array{T}, B))
  CT = Array{T}(size(A,2), size(B,2))
  A_ldiv_B!(CT, AT, BT)
end

# factorization routines

for sfx in ("", "!")
  f = Symbol("pqrfact", sfx)
  g = Symbol("pqrfact_none", sfx)
  h = Symbol("pqr", sfx)
  @eval begin
    function $f{S}(
        trans::Symbol, A::AbstractMatOrLinOp{S}, opts::LRAOptions=LRAOptions(S);
        args...)
      chktrans(trans)
      opts = copy(opts; args...)
      chkopts!(opts, A)
      opts.sketch == :none && return $g(trans, A, opts)
      V = idfact(trans, A, opts)
      F = qrfact!(getcols(trans, A, V[:sk]))
      retq = contains(opts.pqrfact_retval, "q")
      retr = contains(opts.pqrfact_retval, "r")
      rett = contains(opts.pqrfact_retval, "t")
      Q = retq ? full(F[:Q]) : nothing
      R = retr ? pqrr(F[:R], V[:T]) : nothing
      T = rett ? V[:T] : nothing
      retq && retr && !rett && return PartialQR(Q, R, V[:p])
      PQRFactors(Q, R, V[:p], V[:k], T)
    end
    $f(trans::Symbol, A, args...; kwargs...) =
      $f(trans, LinOp(A), args...; kwargs...)
    $f(A, args...; kwargs...) = $f(:n, A, args...; kwargs...)

    function $h(trans::Symbol, A, args...; kwargs...)
      push!(kwargs, (:pqrfact_retval, "qr"))
      F = $f(trans, A, args...; kwargs...)
      F.Q, F.R, F.p
    end
    $h(A, args...; kwargs...) = $h(:n, A, args...; kwargs...)
  end
end

pqrfact_none!(trans::Symbol, A::StridedMatrix, opts::LRAOptions) =
  pqrfact_backend!(trans == :n ? A : A', opts)
pqrfact_none(trans::Symbol, A::StridedMatrix, opts::LRAOptions) =
  pqrfact_backend!(trans == :n ? copy(A) : A', opts)

function pqrr{S}(R::Matrix{S}, T::Matrix{S})
  k, n = size(T)
  n += k
  R_ = Array{S}(k, n)
  R1 = view(R_, :,   1:k)
  R2 = view(R_, :, k+1:n)
  copy!(R1, R)
  copy!(R2, T)
  A_mul_B!(UpperTriangular(R1), R2)
  R_
end

## core backend routine: rank-adaptive GEQP3 with determinant maximization
function pqrfact_backend!(A::StridedMatrix, opts::LRAOptions)
  p, tau, k = geqp3_adap!(A, opts)
  pqrback_postproc(A, p, tau, k, opts)
end

function geqp3_adap!{T}(A::StridedMatrix{T}, opts::LRAOptions)
  m, n = size(A)
  jpvt = collect(BlasInt, 1:n)
  l    = min(m, n)
  k    = (opts.rank < 0 || opts.rank > l) ? l : opts.rank
  tau  = Array{T}(k)
  if k > 0
    k = geqp3_adap_main!(A, jpvt, tau, opts)
  end
  jpvt = convert(Array{Int}, jpvt)
  jpvt, tau, k
end

function geqp3_adap_main!{T<:BlasFloat}(
    A::StridedMatrix{T}, jpvt::Vector{BlasInt}, tau::Vector{T},
    opts::LRAOptions)
  chkstride1(A)
  lda = stride(A, 2)
  n   = length(jpvt)
  k   = length(tau)

  # set block size and allocate work array
  nb      = min(opts.nb, k)
  is_real = T <: Real
  lwork   = 2*n*is_real + (n + 1)*nb
  work    = Array{T}(lwork)

  # initialize column norms
  if is_real
    @inbounds for j = 1:n
      work[j] = work[n+j] = norm(view(A,:,j))
    end
  else
    rwork = Array{eltype(real(zero(T)))}(2*n)
    @inbounds for j = 1:n
      rwork[j] = rwork[n+j] = norm(view(A,:,j))
    end
  end
  maxnrm = maximum(view(is_real ? work : rwork, 1:n))

  # set pivot threshold
  ptol = max(opts.atol, opts.rtol*maxnrm)

  # block factorization
  j = 1
  fjb = Ref{BlasInt}()
  while j <= k
    jb = BlasInt(min(nb, k-j+1))
    if is_real
      _LAPACK.laqps!(
        BlasInt(j-1), jb, fjb, view(A,:,j:n), view(jpvt,j:n), view(tau,j:k),
        view(work,j:n), view(work,n+j:2*n),
        view(work,2*n+1:2*n+nb), view(work,2*n+jb+1:lwork))
    else
      _LAPACK.laqps!(
        BlasInt(j-1), jb, fjb, view(A,:,j:n), view(jpvt,j:n), view(tau,j:k),
        view(rwork,j:n), view(rwork,n+j:2*n),
        view(work,1:nb), view(work,jb+1:lwork))
    end
    jn = j + fjb[]

    # check for rank termination
    if abs(A[jn-1,jn-1]) <= ptol
      @inbounds for i = j:jn-1
        abs(A[i,i]) <= ptol && return i - 1
      end
    end
    j = jn
  end
  k
end

function pqrback_postproc{S}(
    A::StridedMatrix{S}, p::Vector{Int}, tau::Vector{S}, k::Integer,
    opts::LRAOptions)
  retq = contains(opts.pqrfact_retval, "q")
  retr = contains(opts.pqrfact_retval, "r")
  rett = contains(opts.pqrfact_retval, "t")
  maxdet = 0 < k < size(A,2) && opts.maxdet_tol >= 0
  Q = retq ? LAPACK.orgqr!(A[:,1:k], tau, k) : nothing
  R = retr || rett || maxdet ? triu!(A[1:k,:]) : nothing
  T = rett || maxdet ? maxdet_t(R) : nothing
  if maxdet
    maxdet_swapcols!(Q, R, p, T, opts)
    R = retr ? R : nothing
  end
  retq && retr && !rett && return PartialQR(Q, R, p)
  PQRFactors(Q, R, p, k, T)
end

function maxdet_t{S}(R::StridedMatrix{S})
  k, n = size(R)
  T = R[:,k+1:n]
  A_ldiv_B!(UpperTriangular(view(R,1:k,1:k)), T)
end

## rank-revealing QR determinant maximization
function maxdet_swapcols!{S}(
    Q::Union{Matrix{S}, Void}, R::Matrix{S}, p::Vector{Int}, T::Matrix{S},
    opts::LRAOptions)
  k, n  = size(R)
  R1    = view(R, :, 1:k)
  work  = Array{S}(max(n, 2*k))
  retq  = contains(opts.pqrfact_retval, "q")
  retr  = contains(opts.pqrfact_retval, "r")
  niter = 0
  while true
    Tmax, idx = findmaxabs(T)
    Tmax <= 1 + opts.maxdet_tol && break
    if niter == opts.maxdet_niter
      opts.verb &&
        warn("iteration limit ($niter) reached in determinant maximization")
      break
    end
    niter += 1
    i, j = ind2sub((k, n-k), idx)
    maxdet_update!(R1, p, T, work, i, j, retr)
  end
  niter == 0 && return
  F = qrfact!(R1)
  retq && LAPACK.gemqrt!('R', 'N', F.factors, F.T, Q)
  if retr
    triu!(R1)
    R2 = view(R, :, k+1:n)
    copy!(R2, T)
    A_mul_B!(UpperTriangular(R1), R2)
  end
end

## column swap update based on Sherman-Morrison
function maxdet_update!{S}(
    R1::StridedMatrix{S}, p::Vector{Int}, T::Matrix{S}, work::Vector{S},
    i::Integer, j::Integer, retr::Bool)
  k, n = size(T)
  n += k
  p[i], p[k+j] = p[k+j], p[i]
  @inbounds @simd for l = 1:k
    work[l] = T[l,j]
    T[l,j]  = 0
  end
  work[i] -= 1
  T[i,j]   = 1
  @inbounds for l = 1:n-k
    work[k+l] = conj(T[i,l])
  end
  BLAS.ger!(-1/(1 + work[i]), view(work,1:k), view(work,k+1:n), T)
  if retr
    A_mul_B!(view(work,k+1:2*k), R1, view(work,1:k))
    @inbounds @simd for l = 1:k
      R1[l,i] += work[k+l]
    end
  end
end
