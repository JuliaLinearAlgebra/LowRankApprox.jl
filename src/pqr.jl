#= src/pqr.jl

References:

  P. Businger, G.H. Golub. Linear least squares solutions by Householder
    transformations. Numer. Math. 7: 269-276, 1965.

  M. Gu, S.C. Eisenstat. Efficient algorithms for computing a strong
    rank-revealing QR factorization. SIAM J. Sci. Comput. 17 (4): 848-869, 1996.
=#

# PartialQRFactors

type PartialQRFactors
  Q::Union{Matrix, Void}
  R::Union{Matrix, Void}
  p::Vector{Int}
  k::Int
  T::Union{Matrix, Void}
end
typealias PQRFactors PartialQRFactors

function getindex(A::PQRFactors, d::Symbol)
  if     d == :P  return ColumnPermutation(A.p)
  elseif d == :Q  return A.Q
  elseif d == :R  return isa(A.R, Void) ? A.R : UpperTrapezoidal(A.R)
  elseif d == :T  return A.T
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
  f! = symbol(f, "!")
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
  f! = symbol(f, "!")
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
  f!  = symbol(f, "!")
  f!! = symbol(f, "!!")
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
  f! = symbol(f, "!")
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
  h! = symbol(h, "!")
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
      CT = Array(T, size(A,$i))
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
      CT = Array(T, size(A,$i), size(B,$j))
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
      CT = Array(T, size(A,$i), size(B,$j))
      $f!(CT, AT, BT)
    end
  end
end

## left-division
function \{TA,TB}(A::PartialQR{TA}, B::StridedVector{TB})
  T = promote_type(TA, TB)
  AT = convert(PartialQR{T}, A)
  BT = (T == TB ? B : convert(Array{T}, B))
  CT = Array(T, size(A,2))
  A_ldiv_B!(CT, AT, BT)
end
function \{TA,TB}(A::PartialQR{TA}, B::StridedMatrix{TB})
  T = promote_type(TA, TB)
  AT = convert(PartialQR{T}, A)
  BT = (T == TB ? B : convert(Array{T}, B))
  CT = Array(T, size(A,2), size(B,2))
  A_ldiv_B!(CT, AT, BT)
end

# factorization routines

for sfx in ("", "!")
  f = symbol("pqrfact", sfx)
  g = symbol("pqrfact_none", sfx)
  h = symbol("pqr", sfx)
  @eval begin
    function $f(trans::Symbol, A::AbstractMatOrLinOp, opts::LRAOptions; args...)
      opts = isempty(args) ? opts : copy(opts; args...)
      opts = chkopts(A, opts)
      chktrans(trans)
      opts.sketch == :none && return $g(trans, A, opts)
      V = idfact(trans, A, opts)
      F = qrfact!(getcols(trans, A, V[:sk]))
      retq = contains(opts.pqrfact_retval, "q")
      retr = contains(opts.pqrfact_retval, "r")
      rett = contains(opts.pqrfact_retval, "t")
      Q = retq ? full(F[:Q]) : nothing
      R = retr || rett ? [F[:R] UpperTriangular(F[:R])*V[:T]] : nothing
      p = V[:p]
      T = rett ? rrqrt(R) : nothing
      retq && retr && !rett && return PartialQR(Q, R, p)
      PQRFactors(Q, R, p, V[:k], T)
    end
    $f(trans::Symbol, A::AbstractMatOrLinOp; args...) =
      $f(trans, A, LRAOptions(; args...))
    $f(trans::Symbol, A, args...; kwargs...) =
      $f(trans, LinOp(A), args...; kwargs...)
    $f(A, args...; kwargs...) = $f(:n, A, args...; kwargs...)

    function $h(trans::Symbol, A, args...; kwargs...)
      for (index, (key, value)) in enumerate(kwargs)
        if key == :pqrfact_retval && value != "qr"
          warn("keyword \"pqrfact_retval\" ignored")
          kwargs[index] = (key, "qr")
        end
      end
      F = $f(trans, A, args...; kwargs...)
      F.Q, F.R, F.p
    end
    $h(A, args...; kwargs...) = $h(:n, A, args...; kwargs...)
  end
end

pqrfact_none!(trans::Symbol, A::StridedMatrix, opts::LRAOptions) =
  pqrfact_lapack!(trans == :n ? A : A', opts)
pqrfact_none(trans::Symbol, A::StridedMatrix, opts::LRAOptions) =
  pqrfact_lapack!(trans == :n ? copy(A) : A', opts)

## core backend routine: rank-adaptive GEQP3 with RRQR postprocessing
function pqrfact_lapack!{S<:BlasFloat}(A::StridedMatrix{S}, opts::LRAOptions)
  chkstride1(A)
  m, n = size(A)
  lda  = stride(A, 2)
  jpvt = collect(1:n)
  l    = min(m, n)
  k    = (opts.rank < 0 || opts.rank > l) ? l : opts.rank
  tau  = Array(S, k)

  # quick return if empty
  k == 0 && @goto ret

  # set block size and allocate work array
  nb      = min(opts.nb, k)
  is_real = S <: Real
  lwork   = 2*n*is_real + (n + 1)*nb
  work    = Array(S, lwork)

  # initialize column norms
  if is_real
    for j = 1:n
      work[j] = work[n+j] = norm(sub(A,:,j))
    end
  else
    rwork = Array(eltype(real(zero(S))), 2*n)
    for j = 1:n
      rwork[j] = rwork[n+j] = norm(sub(A,:,j))
    end
  end
  maxnrm = maximum(sub(is_real ? work : rwork, 1:n))

  # set pivot threshold
  ptol = max(opts.atol, opts.rtol*maxnrm)

  # block factorization
  j = 1
  fjb = Array(BlasInt, 1)
  while j <= k
    jb = BlasInt(min(nb, k-j+1))
    if is_real
      _LAPACK.laqps!(
        j-1, jb, fjb, sub(A,:,j:n), sub(jpvt,j:n), sub(tau,j:k),
        sub(work,j:n), sub(work,n+j:2*n),
        sub(work,2*n+1:2*n+nb), sub(work,2*n+jb+1:lwork))
    else
      _LAPACK.laqps!(
        j-1, jb, fjb, sub(A,:,j:n), sub(jpvt,j:n), sub(tau,j:k),
        sub(rwork,j:n), sub(rwork,n+j:2*n),
        sub(work,1:nb), sub(work,jb+1:lwork))
    end
    j += fjb[1]

    # check for rank termination
    if abs(A[j-1,j-1]) <= ptol
      for i = (j-fjb[1]):j-1
        if abs(A[i,i]) <= ptol
          k = i - 1
          @goto ret
        end
      end
    end
  end

  @label ret
  retval = lowercase(opts.pqrfact_retval)
  retq = contains(retval, "q")
  retr = contains(retval, "r")
  rett = contains(retval, "t")
  rrqr = k > 0 && opts.rrqr_delta >= 0
  Q = retq ? _LAPACK.orgqr!(A[:,1:k], tau[1:k]) : nothing
  R = retr || rett || rrqr ? triu!(A[1:k,:]) : nothing
  T = rett || rrqr ? rrqrt(R) : nothing

  # RRQR postprocessing
  if rrqr
    qrT, qrwork = _LAPACK.geqrt_init(R, opts)
    niter = 0
    while true
      Tmax, idx = findmax(abs(T))
      Tmax <= 1 + opts.rrqr_delta && break
      if niter == opts.rrqr_niter
        warn("iteration limit ($niter) reached in RRQR postprocessing")
        break
      end
      niter += 1
      i, j = ind2sub((k, n-k), idx)
      swapcols!(R, i, k+j)
      jpvt[i], jpvt[k+j] = jpvt[k+j], jpvt[i]
      F = QRCompactWY(_LAPACK.geqrt!(R, qrT, qrwork)...)
      Q = retq ? Q*F[:Q] : Q
      R = triu!(F.factors)
      rrqrt!(T, R)
    end
  end
  retq && retr && !rett && return PartialQR(Q, R, jpvt)
  PQRFactors(Q, R, jpvt, k, T)
end

function rrqrt!{S}(T::StridedMatrix{S}, R::StridedMatrix{S})
  k, n = size(R)
  size(T) == (k, n-k) || throw(DimensionMismatch)
  copy!(T, sub(R,1:k,k+1:n))
  BLAS.trsm!('L', 'U', 'N', 'N', one(S), sub(R,1:k,1:k), T)
end

function rrqrt{S}(R::StridedMatrix{S})
  k, n = size(R)
  T = Array(S, (k, n-k))
  rrqrt!(T, R)
end