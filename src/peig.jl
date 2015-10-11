#= src/peig.jl

References:

  N. Halko, P.G. Martinsson, J.A. Tropp. Finding structure with randomness:
    Probabilistic algorithms for constructing approximate matrix
    decompositions. SIAM Rev. 53 (2): 217-288, 2011.
=#

abstract AbstractPartialEigen{T} <: Factorization{T}

type PartialEigen{T} <: AbstractPartialEigen{T}
  values::Vector{T}
  vectors::Matrix{T}
end

type HermitianPartialEigen{T,Tr<:Real} <: AbstractPartialEigen{T}
  values::Vector{Tr}
  vectors::Matrix{T}
end
typealias HermPartialEigen HermitianPartialEigen

conj!(A::PartialEigen) = PartialEigen(conj!(A.values), conj!(A.vectors))
conj!(A::HermPartialEigen) = HermPartialEigen(conj!(A.values), conj!(A.vectors))
conj(A::AbstractPartialEigen) = conj!(copy(A))

convert{T}(::Type{PartialEigen{T}}, A::PartialEigen) =
  PartialEigen(convert(Array{T}, A.values), convert(Array{T}, A.vectors))
function convert{T}(::Type{HermPartialEigen{T}}, A::HermPartialEigen)
  Tr = real(T)
  HermPartialEigen(convert(Array{Tr}, A.values), convert(Array{T}, A.vectors))
end
convert{T}(::Type{AbstractPartialEigen{T}}, A::PartialEigen) =
  convert(PartialEigen{T}, A)
convert{T}(::Type{AbstractPartialEigen{T}}, A::HermPartialEigen) =
  convert(HermPartialEigen{T}, A)
convert{T}(::Type{Factorization{T}}, A::PartialEigen) =
  convert(PartialEigen{T}, A)
convert{T}(::Type{Factorization{T}}, A::HermPartialEigen) =
  convert(HermPartialEigen{T}, A)
convert(::Type{Array}, A::HermPartialEigen) = full(A)
convert{T}(::Type{Array{T}}, A::HermPartialEigen) = convert(Array{T}, full(A))

copy(A::PartialEigen) = PartialEigen(copy(A.values), copy(A.vectors))
copy(A::HermPartialEigen) = HermPartialEigen(copy(A.values), copy(A.vectors))

ctranspose!(A::HermPartialEigen) = A
ctranspose(A::HermPartialEigen) = copy(A)
transpose!(A::HermPartialEigen) = conj!(A.vectors)
transpose(A::HermPartialEigen) = HermPartialEigen(A.values, conj(A.vectors))

full(A::HermPartialEigen) = scale(A[:vectors], A[:values])*A[:vectors]'

function getindex(A::AbstractPartialEigen, d::Symbol)
  if     d == :k        return length(A.values)
  elseif d == :values   return A.values
  elseif d == :vectors  return A.vectors
  else                  throw(KeyError(d))
  end
end

ishermitian(A::PartialEigen) = false
issym(A::PartialEigen) = false
ishermitian(A::HermPartialEigen) = true
issym{T}(A::HermPartialEigen{T}) = T <: Real

isreal{T}(A::AbstractPartialEigen{T}) = T <: Real

ndims(A::AbstractPartialEigen) = 2

size(A::AbstractPartialEigen) = (size(A.vectors,1), size(A.vectors,1))
size(A::AbstractPartialEigen, dim::Integer) =
  dim == 1 || dim == 2 ? size(A.vectors,1) : 1

# BLAS/LAPACK multiplication/division routines

## left-multiplication

A_mul_B!{T}(
    y::StridedVector{T}, A::HermPartialEigen{T}, x::StridedVector{T}) =
  A_mul_B!(y, A[:vectors], scalevec!(A[:values], A[:vectors]'*x))
A_mul_B!{T}(
    C::StridedMatrix{T}, A::HermPartialEigen{T}, B::StridedMatrix{T}) =
  A_mul_B!(C, A[:vectors], scale!(A[:values], A[:vectors]'*B))

A_mul_Bc!{T}(C::StridedMatrix{T}, A::HermPartialEigen{T}, B::StridedMatrix{T}) =
  A_mul_B!(C, A[:vectors], scale!(A[:values], A[:vectors]'*B'))
A_mul_Bt!{T<:Real}(
    C::StridedMatrix{T}, A::HermPartialEigen{T}, B::StridedMatrix{T}) =
  A_mul_Bc!(C, A, B)
A_mul_Bt!!{T<:Complex}(
    C::StridedMatrix{T}, A::HermPartialEigen{T}, B::StridedMatrix{T}) =
  A_mul_Bc!(C, A, conj!(B))  # overwrites B
function A_mul_Bt!{T<:Complex}(
    C::StridedMatrix{T}, A::HermPartialEigen{T}, B::StridedMatrix{T})
  size(B, 1) <= A[:k] && return A_mul_Bt!!(C, A, copy(B))
  tmp = (A[:vectors]')*B.'
  scale!(A[:values], tmp)
  A_mul_B!(C, A[:vectors], tmp)
end

Ac_mul_B!{T}(
    C::StridedVecOrMat{T}, A::HermPartialEigen{T}, B::StridedVecOrMat{T}) =
  A_mul_B!(C, A, B)
function At_mul_B!{T}(
    y::StridedVector{T}, A::HermPartialEigen{T}, x::StridedVector{T})
  tmp = A[:vectors].'*x
  scalevec!(A[:values], tmp)
  A_mul_B!(y, A[:vectors], conj!(tmp))
  conj!(y)
end
function At_mul_B!{T}(
    C::StridedMatrix{T}, A::HermPartialEigen{T}, B::StridedMatrix{T})
  tmp = A[:vectors].'*B
  scale!(A[:values], tmp)
  A_mul_B!(C, A[:vectors], conj!(tmp))
  conj!(C)
end

Ac_mul_Bc!{T}(
    C::StridedMatrix{T}, A::HermPartialEigen{T}, B::StridedMatrix{T}) =
  A_mul_Bc!(C, A, B)
function At_mul_Bt!{T}(
    C::StridedMatrix{T}, A::HermPartialEigen{T}, B::StridedMatrix{T})
  tmp = A[:vectors].'*B.'
  scale!(A[:values], tmp)
  A_mul_B!(C, A[:vectors], conj!(tmp))
  conj!(C)
end

## right-multiplication

A_mul_B!{T}(C::StridedMatrix{T}, A::StridedMatrix{T}, B::HermPartialEigen{T}) =
  A_mul_Bc!(C, scale!(A*B[:vectors], B[:values]), B[:vectors])

A_mul_Bc!{T}(C::StridedMatrix{T}, A::StridedMatrix{T}, B::HermPartialEigen{T}) =
  A_mul_B!(C, A, B)
function A_mul_Bt!!{T}(
    C::StridedMatrix{T}, A::StridedMatrix{T}, B::HermPartialEigen{T})
  tmp = conj!(A)*B[:vectors]
  scale!(conj!(tmp), B[:values])
  A_mul_Bt!(C, tmp, B[:vectors])
end  # overwrites A
function A_mul_Bt!{T}(
    C::StridedMatrix{T}, A::StridedMatrix{T}, B::HermPartialEigen{T})
  size(A, 1) <= B[:k] && return A_mul_Bt!!(C, copy(A), B)
  tmp = A*conj(B[:vectors])
  scale!(tmp, B[:values])
  A_mul_Bt!(C, tmp, B[:vectors])
end

for f in (:Ac_mul_B, :At_mul_B)
  f! = symbol(f, "!")
  @eval begin
    function $f!{T}(
        C::StridedMatrix{T}, A::StridedMatrix{T}, B::HermPartialEigen{T})
      tmp = $f(A, B[:vectors])
      scale!(tmp, B[:values])
      A_mul_Bc!(C, tmp, B[:vectors])
    end
  end
end

Ac_mul_Bc!{T}(
    C::StridedMatrix{T}, A::StridedMatrix{T}, B::HermPartialEigen{T}) =
  Ac_mul_B!(C, A, B)
function At_mul_Bt!{T}(
    C::StridedMatrix{T}, A::StridedMatrix{T}, B::HermPartialEigen{T})
  tmp = A'*B[:vectors]
  scale!(conj!(tmp), B[:values])
  A_mul_Bt!(C, tmp, B[:vectors])
end

## left-division (pseudoinverse left-multiplication)
A_ldiv_B!{T}(
    y::StridedVector{T}, A::HermPartialEigen{T}, x::StridedVector{T}) =
  A_mul_B!(y, A[:vectors], iscalevec!(A[:values], A[:vectors]'*x))
A_ldiv_B!{T}(
    C::StridedMatrix{T}, A::HermPartialEigen{T}, B::StridedMatrix{T}) =
  A_mul_B!(C, A[:vectors], iscale!(A[:values], A[:vectors]'*B))

# standard operations

## left-multiplication

for (f, f!, i) in ((:*,        :A_mul_B!,  1),
                   (:Ac_mul_B, :Ac_mul_B!, 2),
                   (:At_mul_B, :At_mul_B!, 2))
  @eval begin
    function $f{TA,TB}(A::HermPartialEigen{TA}, B::StridedVector{TB})
      T = promote_type(TA, TB)
      AT = convert(HermPartialEigen{T}, A)
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
    function $f{TA,TB}(A::HermPartialEigen{TA}, B::StridedMatrix{TB})
      T = promote_type(TA, TB)
      AT = convert(HermPartialEigen{T}, A)
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
    function $f{TA,TB}(A::StridedMatrix{TA}, B::HermPartialEigen{TB})
      T = promote_type(TA, TB)
      AT = (T == TA ? A : convert(Array{T}, A))
      BT = convert(HermPartialEigen{T}, B)
      CT = Array(T, size(A,$i), size(B,$j))
      $f!(CT, AT, BT)
    end
  end
end

## left-division
function \{TA,TB}(A::HermPartialEigen{TA}, B::StridedVector{TB})
  T = promote_type(TA, TB)
  AT = convert(HermPartialEigen{T}, A)
  BT = (T == TB ? B : convert(Array{T}, B))
  CT = Array(T, size(A,2))
  A_ldiv_B!(CT, AT, BT)
end
function \{TA,TB}(A::HermPartialEigen{TA}, B::StridedMatrix{TB})
  T = promote_type(TA, TB)
  AT = convert(HermPartialEigen{T}, A)
  BT = (T == TB ? B : convert(Array{T}, B))
  CT = Array(T, size(A,2), size(B,2))
  A_ldiv_B!(CT, AT, BT)
end

# factorization routines

function peigfact{T}(A::AbstractMatOrLinOp{T}, opts::LRAOptions)
  chksquare(A)
  if ishermitian(A)
    V = idfact(:n, A, opts)
    F = qrfact!(full(V)')
    Q = F[:Q]
    B = F[:R]*(A[V[:sk],V[:sk]]*F[:R]')
    n = size(B, 2)
    for i = 1:n
      B[i,i] = real(B[i,i])
    end
    F = eigfact!(Hermitian(B))
    kn, kp = peigrank_herm(F[:values], opts)
    if kn + kp < n
      idx = [1:kn; n-kp+1:n]
      return HermPartialEigen(F.values[idx], Q*F.vectors[:,idx])
    end
    return HermPartialEigen(F.values, Q*F.vectors)
  else
    # quick return if empty
    size(A,2) == 0 && return PartialEigen(zeros(T, 0), zeros(T, 0, 0))

    # compress with uniform basis
    U = curfact(A, opts)
    F = CUR(A, U)
    kc = F[:kc]
    kr = F[:kr]
    B = Array(T, size(A,1), kc+kr)
    copy!(sub(B,:,1:kc), F[:C])
    ctranspose!(sub(B,:,kc+1:kc+kr), F[:R])
    Q = pqrfact_lapack!(B, copy(opts, rank=2*opts.rank))[:Q]
    B = Q'*(A*Q)
    n = size(B, 2)

    # set job options
    if opts.peig_vecs == :left
      jobvl = 'V'
      jobvr = 'N'
    else
      jobvl = 'N'
      jobvr = 'V'
    end

    if T <: Real
      WR, WI, VL, VR = LAPACK.geev!(jobvl, jobvr, B)
      V = (jobvl == 'V' ? VL : VR)

      # find rank
      k = peigrank(WR, WI, opts)

      # real eigenvectors
      if all(WI .== 0)
        k < n && return PartialEigen(WR[1:k], Q*sub(V,:,1:k))
        return PartialEigen(WR, Q*V)
      end

      # complex eigenvectors
      evec = zeros(Complex{T}, n, k)
      j = 1
      while j <= k
        if WI[j] == 0
          copy!(sub(evec,:,j), sub(V,:,j))
        else
          a = sub(V, :, j  )
          b = sub(V, :, j+1)
          evec[:,j  ] = a + b*im
          evec[:,j+1] = a - b*im
          j += 1
        end
        j += 1
      end
      return PartialEigen(complex(WR[1:k], WI[1:k]), Q*sub(evec,:,1:k))
    else
      W, VL, VR = LAPACK.geev!(jobvl, jobvr, B)
      V = (jobvl == 'V' ? VL : VR)
      k = peigrank(W, opts)
      k < n && return PartialEigen(W[1:k], Q*sub(V,:,1:k))
      return PartialEigen(W, Q*V)
    end
  end
end
peig(A, args...) = (F = peigfact(A, args...); (F.values, F.vectors))

function peigvals{T}(A::AbstractMatOrLinOp{T}, opts::LRAOptions)
  chksquare(A)
  if ishermitian(A)
    V = idfact(:n, A, opts)
    F = qrfact!(full(V)')
    Q = F[:Q]
    B = F[:R]*(A[V[:sk],V[:sk]]*F[:R]')
    n = size(B, 2)
    for i = 1:n
      B[i,i] = real(B[i,i])
    end
    v = eigvals!(Hermitian(B))
    kn, kp = peigrank_herm(v, opts)
    kn + kp < n && return v[[1:kn; n-kp+1:n]]
  else
    U = curfact(A, opts)
    F = CUR(A, U)
    kc = F[:kc]
    kr = F[:kr]
    B = Array(T, size(A,1), kc+kr)
    copy!(sub(B,:,1:kc), F[:C])
    ctranspose!(sub(B,:,kc+1:kc+kr), F[:R])
    Q = pqrfact_lapack!(B, copy(opts, rank=2*opts.rank))[:Q]
    v = eigvals!(Q'*(A*Q))
    k = peigrank(v, opts)
    k < length(v) && return v[1:k]
  end
  v
end

for f in (:peigfact, :peigvals)
  @eval begin
    function $f(A::AbstractMatOrLinOp, rank_or_rtol::Real)
      opts = (rank_or_rtol < 1 ? LRAOptions(rtol=rank_or_rtol)
                               : LRAOptions(rank=rank_or_rtol))
      $f(A, opts)
    end
    $f{T}(A::AbstractMatOrLinOp{T}) = $f(A, default_rtol(T))
    $f(A, args...) = $f(LinOp(A), args...)
  end
end

function peigrank{T<:Real}(wr::Vector{T}, wi::Vector{T}, opts::LRAOptions)
  k = length(wr)
  k = opts.rank >= 0 ? min(opts.rank, k) : k
  ptol = max(opts.atol, opts.rtol*abs(complex(wr[1], wi[1])))
  for i = 2:k
    abs(complex(wr[i], wi[i])) <= ptol && return i - 1
  end
  k
end
function peigrank{T<:Real}(
    w::StridedVector, opts::LRAOptions; pivot::T=abs(w[1]))
  k = length(w)
  k = opts.rank >= 0 ? min(opts.rank, k) : k
  ptol = max(opts.atol, opts.rtol*pivot)
  for i = 2:k
    abs(w[i]) <= ptol && return i - 1
  end
  k
end
function peigrank_herm{T<:Real}(w::Vector{T}, opts::LRAOptions)
  n = length(w)
  k = opts.rank >= 0 ? min(opts.rank, n) : n
  pivot = max(abs(w[1]), abs(w[n]))
  idx = searchsorted(w, 0)
  kn = peigrank(sub(w,          1:first(idx)-1), opts, pivot=pivot)
  kp = peigrank(sub(w,last(idx)+1:n)           , opts, pivot=pivot)
  kn, kp
end