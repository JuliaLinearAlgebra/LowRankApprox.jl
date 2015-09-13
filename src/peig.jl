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
  Tr = eltype(real(one(T)))
  HermPartialEigen(convert(Array{Tr}, A.values), convert(Array{T}, A.vectors))
end
convert(::Type{Array}, A::HermPartialEigen) = full(A)
convert{T}(::Type{Array{T}}, A::HermPartialEigen) = convert(Array{T}, full(A))

copy(A::PartialEigen) = PartialEigen(copy(A.values), copy(A.vectors))
copy(A::HermPartialEigen) = HermPartialEigen(copy(A.values), copy(A.vectors))

ctranspose!(A::HermPartialEigen) = A
ctranspose(A::HermPartialEigen) = A
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

# BLAS/LAPACK multiplication routines

## left-multiplication

A_mul_B!{T}(y::StridedVector{T}, A::HermPartialEigen{T}, x::StridedVector{T}) =
  A_mul_B!(y, A[:vectors], scalevec!(A[:values], A[:vectors]'*x))
A_mul_B!{T}(C::StridedMatrix{T}, A::HermPartialEigen{T}, B::StridedMatrix{T}) =
  A_mul_B!(C, A[:vectors], scale!(A[:values], A[:vectors]'*B))

function A_mul_Bc!{T}(
    C::StridedMatrix{T}, A::HermPartialEigen{T}, B::StridedMatrix{T})
  tmp = A[:vectors]'*B'
  scale!(A[:values], tmp)
  A_mul_B!(C, A[:vectors], tmp)
end
A_mul_Bt!{T<:Real}(
    C::StridedMatrix{T}, A::HermPartialEigen{T}, B::StridedMatrix{T}) =
  A_mul_Bc!(C, A, B)
A_mul_Bt!!{T<:Complex}(
    C::StridedMatrix{T}, A::HermPartialEigen{T}, B::StridedMatrix{T}) =
  A_mul_Bc!(C, A, conj!(B))  # overwrites B
A_mul_Bt!{T<:Complex}(
    C::StridedMatrix{T}, A::HermPartialEigen{T}, B::StridedMatrix{T}) =
  A_mul_Bt!!(C, A, copy(B))

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
A_mul_Bt!{T}(C::StridedMatrix{T}, A::StridedMatrix{T}, B::HermPartialEigen{T}) =
  A_mul_Bt!!(C, copy(A), B)

for f in (:Ac_mul_B, :At_mul_B)
  f! = symbol(f, "!")
  @eval begin
    function $f!{T}(C::StridedMatrix{T}, A::StridedMatrix{T}, B::HermPartialEigen{T})
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

# factorization routines

function peigfact{T}(A::AbstractMatOrLinOp{T}, opts::LRAOptions)
  chkopts(opts)
  if ishermitian(A)
    Q = prange(:n, A, opts)
    B = Q'*(A*Q)
    for i = 1:size(B,2)
      B[i,i] = real(B[i,i])
    end
    F = eigfact!(Hermitian(B))
    return HermPartialEigen(F.values, Q*F.vectors)
  else
    n = size(A, 2)

    # quick return if empty
    n == 0 && return PartialEigen(zeros(T, 0), zeros(T, 0, 0))

    # compress with uniform basis
    Q = prange(:nc, A, opts)
    B = Q'*(A*Q)
    k = size(Q, 2)

    # set job options
    if opts.peig_vecs == :left
      jobvl = 'V'
      jobvr = 'N'
    else
      jobvl = 'N'
      jobvr = 'V'
    end

    if T <: Real
      A, WR, WI, VL, VR, _ = LAPACK.geevx!('B', jobvl, jobvr, 'N', B)
      V = (jobvl == 'V' ? VL : VR)

      # find rank
      ptol = max(opts.atol, opts.rtol*abs(complex(WR[1], WI[1])))
      n = k
      for i = 1:k
        abs(complex(WR[i], WI[i])) <= ptol && (k = i - 1; break)
      end

      # real eigenvectors
      all(WI .== 0) && return PartialEigen(WR[1:k], Q*V[:,1:k])

      # complex eigenvectors
      evec = zeros(Complex{T}, n, k)
      j = 1
      while j <= k
        if WI[j] == 0
          evec[:,j] = V[:,j]
        else
          evec[:,j  ] = V[:,j] + im*V[:,j+1]
          evec[:,j+1] = V[:,j] - im*V[:,j+1]
          j += 1
        end
        j += 1
      end
      return PartialEigen(complex(WR[1:k], WI[1:k]), Q*evec[:,1:k])
    else
      A, W, VL, VR, _ = LAPACK.geevx!('B', jobvl, jobvr, 'N', B)

      # find rank
      ptol = max(opts.atol, opts.rtol*abs(W[1]))
      for i = 1:k
        abs(W[i]) <= ptol && (k = i - 1; break)
      end

      return PartialEigen(W[1:k], Q*(jobvl == 'V' ? VL : VR)[:,1:k])
    end
  end
end
peig(A, args...) = (F = peigfact(A, args...); (F.values, F.vectors))

function peigvals(A::AbstractMatOrLinOp, opts::LRAOptions)
  chkopts(opts)
  if ishermitian(A)
    Q = prange(:n, A, opts)
    B = Q'*(A*Q)
    for i = 1:size(B,2)
      B[i,i] = real(B[i,i])
    end
    return eigvals!(Hermitian(B))
  else
    Q = prange(:nc, A, opts)
    w = eigvals!(Q'*(A*Q))
    k = length(w);
    ptol = max(opts.atol, opts.rtol*abs(w[1]))
    for i = 1:k
      abs(w[i]) <= ptol && (k = i - 1; break)
    end
    return w[1:k]
  end
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