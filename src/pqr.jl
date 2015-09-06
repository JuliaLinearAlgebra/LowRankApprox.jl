#= src/pqr.jl

References:

  P. Businger, G.H. Golub. Linear least squares solutions by Householder
    transformations. Numer. Math. 7: 269-276, 1965.

  N. Halko, P.G. Martinsson, J.A. Tropp. Finding structure with randomness:
    Probabilistic algorithms for constructing approximate matrix
    decompositions. SIAM Rev. 53 (2): 217-288, 2011.
=#

type PartialQR{T} <: Factorization{T}
  Q::Matrix{T}
  R::Matrix{T}
  p::Vector{Int}
end

conj!(A::PartialQR) = PartialQR(conj!(A.Q), conj!(A.R), A.p)
conj(A::PartialQR) = conj!(copy(A))

convert{T}(::Type{PartialQR{T}}, A::PartialQR) =
  PartialQR(convert(Array{T}, A.Q), convert(Array{T}, A.R), A.p)
convert(::Type{Array}, A::PartialQR) = full(A)
convert{T}(::Type{Array{T}}, A::PartialQR) = convert(Array{T}, full(A))

copy(A::PartialQR) = PartialQR(copy(A.Q), copy(A.R), copy(A.p))

full(A::PartialQR) = A_mul_Bc!(A[:Q]*A[:R], A[:P])

function getindex{T}(A::PartialQR{T}, d::Symbol)
  if     d == :P  return ColumnPermutation(A.p)
  elseif d == :Q  return A.Q
  elseif d == :R  return UpperTrapezoidal(A.R)
  elseif d == :k  return size(A.Q, 2)
  elseif d == :p  return A.p
  else            throw(KeyError(d))
  end
end

ishermitian(A::PartialQR) = false
issym(A::PartialQR) = false

isreal{T}(A::PartialQR{T}) = T <: Real

ndims(A::PartialQR) = 2

size(A::PartialQR) = (size(A.Q,1), size(A.R,2))
size(A::PartialQR, dim::Integer) =
  dim == 1 ? size(A.Q,1) : (dim == 2 ? size(A.R,2) : 1)

# BLAS/LAPACK multiplication routines

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

# factorization routines

function pqrfact!(A::AbstractMatOrLinOp, opts::LRAOptions)
  if typeof(A) <: StridedMatrix && opts.sketch == :none
    return pqrfact_lapack!(A, opts)
  end
  Q = rrange(:n, A, opts)
  opts = copy(opts, sketch=:none)
  F = pqrfact_lapack!(Q'*A, opts)
  F.Q = Q*F.Q
  F
end
function pqrfact(A::AbstractMatOrLinOp, opts::LRAOptions)
  if typeof(A) <: StridedMatrix && opts.sketch == :none
    return pqrfact!(copy(A), opts)
  end
  pqrfact!(A, opts)
end

for sfx in ("", "!")
  f = symbol("pqrfact", sfx)
  g = symbol("pqr", sfx)
  @eval begin
    function $f(A::AbstractMatOrLinOp, rank_or_rtol::Real)
      opts = (rank_or_rtol < 1 ? LRAOptions(rtol=rank_or_rtol)
                               : LRAOptions(rank=rank_or_rtol))
      $f(A, opts)
    end
    $f{T}(A::AbstractMatOrLinOp{T}) = $f(A, default_rtol(T))
    $f(A, args...) = $f(LinOp(A), args...)

    function $g(A::AbstractMatOrLinOp, args...)
      F = $f(A, args...)
      F.Q, F.R, F.p
    end
  end
end

## core backend routine: GEQP3 with rank termination
function pqrfact_lapack!{T<:BlasFloat}(A::StridedMatrix{T}, opts::LRAOptions)
  chkopts(opts)
  chkstride1(A)
  m, n = size(A)
  lda  = stride(A, 2)
  jpvt = collect(1:n)
  l    = min(m, n)
  k    = (opts.rank < 0 || opts.rank > l) ? l : opts.rank
  tau  = Array(T, k)

  # quick return if empty
  isempty(A) && @goto ret

  # set block size and allocate work array
  nb      = min(opts.nb, k)
  is_real = T <: Real
  lwork   = 2*n*is_real + (n + 1)*nb
  work    = Array(T, lwork)

  # initialize column norms
  if is_real
    for j = 1:n
      work[j] = work[n+j] = norm(sub(A,:,j))
    end
  else
    rwork = Array(eltype(real(zero(T))), 2*n)
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
    jb = min(nb, k-j+1)
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
  Q = _LAPACK.orgqr!(A[:,1:k], tau[1:k])
  R = triu!(A[1:k,:])
  PartialQR(Q, R, jpvt)
end