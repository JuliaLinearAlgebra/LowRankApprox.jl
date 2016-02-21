#= src/pheig.jl
=#

type PartialHermitianEigen{T,Tr<:Real} <: Factorization{T}
  values::Vector{Tr}
  vectors::Matrix{T}
end
typealias PartialHermEigen PartialHermitianEigen

conj!(A::PartialHermEigen) = PartialHermEigen(conj!(A.values), conj!(A.vectors))
conj(A::PartialHermEigen) = conj!(copy(A))

function convert{T}(::Type{PartialHermEigen{T}}, A::PartialHermEigen)
  Tr = real(T)
  PartialHermEigen(convert(Array{Tr}, A.values), convert(Array{T}, A.vectors))
end
convert{T}(::Type{Factorization{T}}, A::PartialHermEigen) =
  convert(PartialHermEigen{T}, A)
convert(::Type{Array}, A::PartialHermEigen) = full(A)
convert{T}(::Type{Array{T}}, A::PartialHermEigen) = convert(Array{T}, full(A))

copy(A::PartialHermEigen) = PartialHermEigen(copy(A.values), copy(A.vectors))

ctranspose!(A::PartialHermEigen) = A
ctranspose(A::PartialHermEigen) = copy(A)
transpose!(A::PartialHermEigen) = conj!(A.vectors)
transpose(A::PartialHermEigen) = PartialHermEigen(A.values, conj(A.vectors))

full(A::PartialHermEigen) = scale(A[:vectors], A[:values])*A[:vectors]'

function getindex(A::PartialHermEigen, d::Symbol)
  if     d == :k        return length(A.values)
  elseif d == :values   return A.values
  elseif d == :vectors  return A.vectors
  else                  throw(KeyError(d))
  end
end

ishermitian(::PartialHermEigen) = true
issym{T}(A::PartialHermEigen{T}) = isreal(A)

isreal{T}(::PartialHermEigen{T}) = T <: Real

ndims(::PartialHermEigen) = 2

size(A::PartialHermEigen) = (size(A.vectors,1), size(A.vectors,1))
size(A::PartialHermEigen, dim::Integer) =
  dim == 1 || dim == 2 ? size(A.vectors,1) : 1

# BLAS/LAPACK multiplication/division routines

## left-multiplication

A_mul_B!{T}(
    y::StridedVector{T}, A::PartialHermEigen{T}, x::StridedVector{T}) =
  A_mul_B!(y, A[:vectors], scalevec!(A[:values], A[:vectors]'*x))
A_mul_B!{T}(
    C::StridedMatrix{T}, A::PartialHermEigen{T}, B::StridedMatrix{T}) =
  A_mul_B!(C, A[:vectors], scale!(A[:values], A[:vectors]'*B))

A_mul_Bc!{T}(C::StridedMatrix{T}, A::PartialHermEigen{T}, B::StridedMatrix{T}) =
  A_mul_B!(C, A[:vectors], scale!(A[:values], A[:vectors]'*B'))
A_mul_Bt!{T<:Real}(
    C::StridedMatrix{T}, A::PartialHermEigen{T}, B::StridedMatrix{T}) =
  A_mul_Bc!(C, A, B)
A_mul_Bt!!{T<:Complex}(
    C::StridedMatrix{T}, A::PartialHermEigen{T}, B::StridedMatrix{T}) =
  A_mul_Bc!(C, A, conj!(B))  # overwrites B
function A_mul_Bt!{T<:Complex}(
    C::StridedMatrix{T}, A::PartialHermEigen{T}, B::StridedMatrix{T})
  size(B, 1) <= A[:k] && return A_mul_Bt!!(C, A, copy(B))
  tmp = (A[:vectors]')*B.'
  scale!(A[:values], tmp)
  A_mul_B!(C, A[:vectors], tmp)
end

Ac_mul_B!{T}(
    C::StridedVecOrMat{T}, A::PartialHermEigen{T}, B::StridedVecOrMat{T}) =
  A_mul_B!(C, A, B)
function At_mul_B!{T}(
    y::StridedVector{T}, A::PartialHermEigen{T}, x::StridedVector{T})
  tmp = A[:vectors].'*x
  scalevec!(A[:values], tmp)
  A_mul_B!(y, A[:vectors], conj!(tmp))
  conj!(y)
end
function At_mul_B!{T}(
    C::StridedMatrix{T}, A::PartialHermEigen{T}, B::StridedMatrix{T})
  tmp = A[:vectors].'*B
  scale!(A[:values], tmp)
  A_mul_B!(C, A[:vectors], conj!(tmp))
  conj!(C)
end

Ac_mul_Bc!{T}(
    C::StridedMatrix{T}, A::PartialHermEigen{T}, B::StridedMatrix{T}) =
  A_mul_Bc!(C, A, B)
function At_mul_Bt!{T}(
    C::StridedMatrix{T}, A::PartialHermEigen{T}, B::StridedMatrix{T})
  tmp = A[:vectors].'*B.'
  scale!(A[:values], tmp)
  A_mul_B!(C, A[:vectors], conj!(tmp))
  conj!(C)
end

## right-multiplication

A_mul_B!{T}(C::StridedMatrix{T}, A::StridedMatrix{T}, B::PartialHermEigen{T}) =
  A_mul_Bc!(C, scale!(A*B[:vectors], B[:values]), B[:vectors])

A_mul_Bc!{T}(C::StridedMatrix{T}, A::StridedMatrix{T}, B::PartialHermEigen{T}) =
  A_mul_B!(C, A, B)
function A_mul_Bt!!{T}(
    C::StridedMatrix{T}, A::StridedMatrix{T}, B::PartialHermEigen{T})
  tmp = conj!(A)*B[:vectors]
  scale!(conj!(tmp), B[:values])
  A_mul_Bt!(C, tmp, B[:vectors])
end  # overwrites A
function A_mul_Bt!{T}(
    C::StridedMatrix{T}, A::StridedMatrix{T}, B::PartialHermEigen{T})
  size(A, 1) <= B[:k] && return A_mul_Bt!!(C, copy(A), B)
  tmp = A*conj(B[:vectors])
  scale!(tmp, B[:values])
  A_mul_Bt!(C, tmp, B[:vectors])
end

for f in (:Ac_mul_B, :At_mul_B)
  f! = symbol(f, "!")
  @eval begin
    function $f!{T}(
        C::StridedMatrix{T}, A::StridedMatrix{T}, B::PartialHermEigen{T})
      tmp = $f(A, B[:vectors])
      scale!(tmp, B[:values])
      A_mul_Bc!(C, tmp, B[:vectors])
    end
  end
end

Ac_mul_Bc!{T}(
    C::StridedMatrix{T}, A::StridedMatrix{T}, B::PartialHermEigen{T}) =
  Ac_mul_B!(C, A, B)
function At_mul_Bt!{T}(
    C::StridedMatrix{T}, A::StridedMatrix{T}, B::PartialHermEigen{T})
  tmp = A'*B[:vectors]
  scale!(conj!(tmp), B[:values])
  A_mul_Bt!(C, tmp, B[:vectors])
end

## left-division (pseudoinverse left-multiplication)
A_ldiv_B!{T}(
    y::StridedVector{T}, A::PartialHermEigen{T}, x::StridedVector{T}) =
  A_mul_B!(y, A[:vectors], iscalevec!(A[:values], A[:vectors]'*x))
A_ldiv_B!{T}(
    C::StridedMatrix{T}, A::PartialHermEigen{T}, B::StridedMatrix{T}) =
  A_mul_B!(C, A[:vectors], iscale!(A[:values], A[:vectors]'*B))

# standard operations

## left-multiplication

for (f, f!, i) in ((:*,        :A_mul_B!,  1),
                   (:Ac_mul_B, :Ac_mul_B!, 2),
                   (:At_mul_B, :At_mul_B!, 2))
  @eval begin
    function $f{TA,TB}(A::PartialHermEigen{TA}, B::StridedVector{TB})
      T = promote_type(TA, TB)
      AT = convert(PartialHermEigen{T}, A)
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
    function $f{TA,TB}(A::PartialHermEigen{TA}, B::StridedMatrix{TB})
      T = promote_type(TA, TB)
      AT = convert(PartialHermEigen{T}, A)
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
    function $f{TA,TB}(A::StridedMatrix{TA}, B::PartialHermEigen{TB})
      T = promote_type(TA, TB)
      AT = (T == TA ? A : convert(Array{T}, A))
      BT = convert(PartialHermEigen{T}, B)
      CT = Array(T, size(A,$i), size(B,$j))
      $f!(CT, AT, BT)
    end
  end
end

## left-division
function \{TA,TB}(A::PartialHermEigen{TA}, B::StridedVector{TB})
  T = promote_type(TA, TB)
  AT = convert(PartialHermEigen{T}, A)
  BT = (T == TB ? B : convert(Array{T}, B))
  CT = Array(T, size(A,2))
  A_ldiv_B!(CT, AT, BT)
end
function \{TA,TB}(A::PartialHermEigen{TA}, B::StridedMatrix{TB})
  T = promote_type(TA, TB)
  AT = convert(PartialHermEigen{T}, A)
  BT = (T == TB ? B : convert(Array{T}, B))
  CT = Array(T, size(A,2), size(B,2))
  A_ldiv_B!(CT, AT, BT)
end

# factorization routines

function pheigfact{T}(
    A::AbstractMatOrLinOp{T}, opts::LRAOptions=LRAOptions(T); args...)
  chksquare(A)
  !ishermitian(A) && error("matrix must be Hermitian")
  opts = isempty(args) ? opts : copy(opts; args...)
  V = idfact(:n, A, opts)
  F = qrfact!(full(:c, V))
  Q = F[:Q]
  B = hermitianize!(F[:R]*(A[V[:sk],V[:sk]]*F[:R]'))
  F = eigfact!(B)
  F = PartialHermEigen(F.values, F.vectors)
  kn, kp = pheigrank(F[:values], opts)
  n = size(B, 2)
  if kn + kp < n
    idx = [1:kn; n-kp+1:n]
    F.values  = F.values[idx]
    F.vectors = F.vectors[:,idx]
  end
  pheigorth!(F.values, F.vectors, opts)
  F.vectors = Q*F.vectors
  F
end

function pheigvals{T}(
    A::AbstractMatOrLinOp{T}, opts::LRAOptions=LRAOptions(T); args...)
  chksquare(A)
  !ishermitian(A) && error("matrix must be Hermitian")
  opts = isempty(args) ? opts : copy(opts; args...)
  V = idfact(:n, A, opts)
  F = qrfact!(full(:c, V))
  B = hermitianize!(F[:R]*(A[V[:sk],V[:sk]]*F[:R]'))
  v = eigvals!(B)
  kn, kp = pheigrank(v, opts)
  n = size(B, 2)
  kn + kp < n && return v[[1:kn; n-kp+1:n]]
  v
end

for f in (:pheigfact, :pheigvals)
  @eval $f(A, args...; kwargs...) = $f(LinOp(A), args...; kwargs...)
end

function pheig(A, args...; kwargs...)
  F = pheigfact(A, args...; kwargs...)
  F.values, F.vectors
end

function pheigrank{T<:Real}(w::Vector{T}, opts::LRAOptions)
  n = length(w)
  k = opts.rank >= 0 ? min(opts.rank, n) : n
  wmax = max(abs(w[1]), abs(w[n]))
  idx = searchsorted(w, 0)
  kn = pheigrank1(sub(w,1:first(idx)-1),   opts, wmax)
  kp = pheigrank1(sub(w,n:-1:last(idx)+1), opts, wmax)
  kn, kp
end
function pheigrank1{T<:Real}(w::StridedVector, opts::LRAOptions, wmax::T)
  k = length(w)
  k = opts.rank >= 0 ? min(opts.rank, k) : k
  ptol = max(opts.atol, opts.rtol*wmax)
  for i = 2:k
    abs(w[i]) <= ptol && return i - 1
  end
  k
end

function pheigorth!{T<:Real}(
    values::Vector{T}, vectors::Matrix, opts::LRAOptions)
  n = length(values)
  a = 1
  while a <= n
    va = values[a]
    b  = a + 1
    while b <= n
      vb = values[b]
      2*abs((va - vb)/(va + vb)) > opts.pheig_orthtol && break
      b += 1
    end
    b -= 1
    for i = a:b
      vi = sub(vectors, :, i)
      for j = i+1:b
        vj = sub(vectors, :, j)
        BLAS.axpy!(-dot(vi,vj), vi, vj)
      end
    end
    a = b + 1
  end
end