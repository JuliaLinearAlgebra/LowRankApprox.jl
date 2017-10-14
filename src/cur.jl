#= src/cur.jl

References:

  S.A. Goreinov, E.E. Tyrtyshnikov, N.L. Zamarashkin. A theory of pseudoskeleton
    approximations. Linear Algebra Appl. 261: 1-21, 1997.

  S. Voronin, P.-G. Martinsson. Efficient algorithms for CUR and interpolative
    matrix decompositions. Preprint, arXiv:1412.8447 [math.NA].
=#

# CURPackedU

abstract type AbstractCURPackedU <: Factorization{Int} end

mutable struct CURPackedU <: AbstractCURPackedU
  rows::Vector{Int}
  cols::Vector{Int}
end

mutable struct HermitianCURPackedU <: AbstractCURPackedU
  cols::Vector{Int}
end
const HermCURPackedU = HermitianCURPackedU

mutable struct SymmetricCURPackedU <: AbstractCURPackedU
  cols::Vector{Int}
end
const SymCURPackedU = SymmetricCURPackedU

const HermOrSymCURPackedU = Union{HermCURPackedU, SymCURPackedU}

copy(A::CURPackedU) = CURPackedU(copy(rows), copy(cols))
copy(A::SymCURPackedU) = SymCURPackedU(copy(cols))
copy(A::HermCURPackedU) = HermCURPackedU(copy(cols))

function getindex(A::CURPackedU, d::Symbol)
  if     d == :cols  return A.cols
  elseif d == :k     return length(A.cols)
  elseif d == :rows  return A.rows
  else               throw(KeyError(d))
  end
end
function getindex(A::HermOrSymCURPackedU, d::Symbol)
  if     d in (:cols, :rows)  return A.cols
  elseif d == :k              return length(A.cols)
  else                        throw(KeyError(d))
  end
end

ndims(A::AbstractCURPackedU) = 2

size(A::AbstractCURPackedU) = (length(A[:cols]), length(A[:rows]))
size(A::AbstractCURPackedU, dim::Integer) =
  (dim == 1 ? length(A[:cols]) : (dim == 2 ? length(A[:rows]) : 1))

# CUR

abstract type AbstractCUR{T} <: Factorization{T} end

mutable struct CUR{T} <: AbstractCUR{T}
  rows::Vector{Int}
  cols::Vector{Int}
  C::Matrix{T}
  U::Factorization{T}
  R::Matrix{T}
end

mutable struct HermitianCUR{T} <: AbstractCUR{T}
  cols::Vector{Int}
  C::Matrix{T}
  U::Factorization{T}
end
const HermCUR = HermitianCUR

mutable struct SymmetricCUR{T} <: AbstractCUR{T}
  cols::Vector{Int}
  C::Matrix{T}
  U::Factorization{T}
end
const SymCUR = SymmetricCUR

const HermOrSymCUR{T} = Union{HermCUR{T}, SymCUR{T}}

function CUR(A::AbstractMatOrLinOp, U::CURPackedU)
  rows = U[:rows]
  cols = U[:cols]
  C = A[:,cols]
  R = A[rows,:]
  F = svdfact!(C[rows,:])
  U = PartialSVD(F[:V], 1 ./ F[:S], F[:U]')
  CUR(rows, cols, C, U, R)
end
CUR(A::AbstractMatOrLinOp, U::HermCURPackedU) = HermCUR(A, U)
CUR(A::AbstractMatOrLinOp, U::SymCURPackedU) = SymCUR(A, U)
CUR(A::AbstractMatOrLinOp, rows, cols) = CUR(A, CURPackedU(rows, cols))
CUR(A, args...) = CUR(LinOp(A), args...)

function HermCUR(A::AbstractMatOrLinOp, U::HermCURPackedU)
  cols = U[:cols]
  C = A[:,cols]
  F = eigfact!(Hermitian(C[cols,:]))
  U = PartialHermEigen(1 ./ F[:values], F[:vectors])
  HermCUR(cols, C, U)
end
HermCUR(A::AbstractMatOrLinOp, cols) = HermCUR(A, HermCURPackedU(cols))
HermCUR(A, args...) = HermCUR(LinOp(A), args...)

function SymCUR(A::AbstractMatOrLinOp, U::SymCURPackedU)
  cols = U[:cols]
  C = A[:,cols]
  F = svdfact!(C[cols,:])
  U = PartialSVD(F[:V], 1 ./ F[:S], F[:U]')
  SymCUR(cols, C, U)
end
SymCUR(A::AbstractMatOrLinOp, cols) = SymCUR(A, SymCURPackedU(cols))
SymCUR(A, args...) = SymCUR(LinOp(A), args...)

conj!(A::CUR) = CUR(A.rows, A.cols, conj!(A.C), conj!(A.U), conj!(A.R))
conj(A::CUR) = CUR(A.rows, A.cols, conj(A.C), conj(A.U), conj(A.R))
conj!(A::HermCUR) = HermCUR(A.cols, conj!(A.C), conj!(A.U))
conj(A::HermCUR) = HermCUR(A.cols, conj(A.C), conj(A.U))
conj!(A::SymCUR) = SymCUR(A.cols, conj!(A.C), conj!(A.U))
conj(A::SymCUR) = SymCUR(A.cols, conj(A.C), conj(A.U))

convert(::Type{CUR{T}}, A::CUR) where {T} =
  CUR(A.rows, A.cols, convert(Matrix{T}, A.C), convert(Factorization{T}, A.U),
      convert(Matrix{T}, A.R))
convert(::Type{HermCUR{T}}, A::HermCUR) where {T} =
  HermCUR(A.cols, convert(Matrix{T}, A.C), convert(Factorization{T}, A.U))
convert(::Type{SymCUR{T}}, A::SymCUR) where {T} =
  SymCUR(A.cols, convert(Matrix{T}, A.C), convert(Factorization{T}, A.U))
convert(::Type{AbstractCUR{T}}, A::CUR) where {T} = convert(CUR{T}, A)
convert(::Type{AbstractCUR{T}}, A::HermCUR) where {T} = convert(HermCUR{T}, A)
convert(::Type{AbstractCUR{T}}, A::SymCUR) where {T} = convert(SymCUR{T}, A)
convert(::Type{Factorization{T}}, A::CUR) where {T} = convert(CUR{T}, A)
convert(::Type{Factorization{T}}, A::HermCUR) where {T} = convert(HermCUR{T}, A)
convert(::Type{Factorization{T}}, A::SymCUR) where {T} = convert(SymCUR{T}, A)
convert(::Type{Array}, A::AbstractCUR) = full(A)
convert(::Type{Array{T}}, A::AbstractCUR) where {T} = convert(Array{T}, full(A))

copy(A::CUR) = CUR(copy(A.rows), copy(A.cols), copy(A.C), copy(A.U), copy(A.R))
copy(A::HermCUR) = HermCUR(copy(A.cols), copy(A.C), copy(A.U))
copy(A::SymCUR) = SymCUR(copy(A.cols), copy(A.C), copy(A.U))

full(A::CUR{T}) where {T} = A[:C]*(A[:U]*A[:R])
full(A::HermCUR{T}) where {T} = A[:C]*(A[:U]*A[:C]')
full(A::SymCUR{T}) where {T} = A[:C]*(A[:U]*A[:C].')

function getindex(A::CUR{T}, d::Symbol) where T
  if     d == :C     return A.C
  elseif d == :R     return A.R
  elseif d == :U     return A.U
  elseif d == :cols  return A.cols
  elseif d == :k     return length(A.cols)
  elseif d == :rows  return A.rows
  else               throw(KeyError(d))
  end
end
function getindex(A::HermCUR{T}, d::Symbol) where T
  if     d == :C              return A.C
  elseif d == :R              return A.C'
  elseif d == :U              return A.U
  elseif d in (:cols, :rows)  return A.cols
  elseif d == :k              return length(A.cols)
  else                        throw(KeyError(d))
  end
end
function getindex(A::SymCUR{T}, d::Symbol) where T
  if     d == :C              return A.C
  elseif d == :R              return A.C.'
  elseif d == :U              return A.U
  elseif d in (:cols, :rows)  return A.cols
  elseif d == :k              return length(A.cols)
  else                        throw(KeyError(d))
  end
end

ishermitian(::CUR) = false
issymmetric(::CUR) = false
ishermitian(::HermCUR) = true
issymnetric(A::HermCUR{T}) where {T} = isreal(A)
ishermitian(A::SymCUR{T}) where {T} = isreal(A)
issymmetric(::SymCUR) = true

isreal(::AbstractCUR{T}) where {T} = T <: Real

ndims(A::AbstractCUR) = 2

size(A::CUR) = (size(A.C,1), size(A.R,2))
size(A::CUR, dim::Integer) =
  (dim == 1 ? size(A.C,1) : (dim == 2 ? size(A.R,2) : 1))
size(A::HermOrSymCUR) = (size(A.C,1), size(A.C,1))
size(A::HermOrSymCUR, dim::Integer) = dim == 1 || dim == 2 ? size(A.C,1) : 1

# BLAS/LAPACK multiplication routines

## CUR left-multiplication

A_mul_B!(C::StridedVecOrMat{T}, A::CUR{T}, B::StridedVecOrMat{T}) where {T} =
  A_mul_B!(C, A[:C], A[:U]*(A[:R]*B))

for f in (:A_mul_Bc, :A_mul_Bt)
  f! = Symbol(f, "!")
  @eval begin
    function $f!(C::StridedMatrix{T}, A::CUR{T}, B::StridedMatrix{T}) where T
      tmp = $f(A[:R], B)
      A_mul_B!(C, A[:C], A[:U]*tmp)
    end
  end
end

for f in (:Ac_mul_B, :At_mul_B)
  f! = Symbol(f, "!")
  @eval begin
    function $f!(C::StridedVecOrMat{T}, A::CUR{T}, B::StridedVecOrMat{T}) where T
      tmp = $f(A[:C], B)
      tmp = $f(A[:U], tmp)
      $f!(C, A[:R], tmp)
    end
  end
end

for (f, g) in ((:Ac_mul_Bc, :Ac_mul_B), (:At_mul_Bt, :At_mul_B))
  f! = Symbol(f, "!")
  g! = Symbol(g, "!")
  @eval begin
    function $f!(C::StridedMatrix{T}, A::CUR{T}, B::StridedMatrix{T}) where T
      tmp = $f(A[:C], B)
      tmp = $g(A[:U], tmp)
      $g!(C, A[:R], tmp)
    end
  end
end

## CUR right-multiplication

A_mul_B!(C::StridedMatrix{T}, A::StridedMatrix{T}, B::CUR{T}) where {T} =
  A_mul_B!(C, (A*B[:C])*B[:U], B[:R])

for f in (:A_mul_Bc, :A_mul_Bt)
  f! = Symbol(f, "!")
  @eval begin
    function $f!(C::StridedMatrix{T}, A::StridedMatrix{T}, B::CUR{T}) where T
      tmp = $f(A, B[:R])
      tmp = $f(tmp, B[:U])
      $f!(C, tmp, B[:C])
    end
  end
end

for f in (:Ac_mul_B, :At_mul_B)
  f! = Symbol(f, "!")
  @eval begin
    function $f!(C::StridedMatrix{T}, A::StridedMatrix{T}, B::CUR{T}) where T
      tmp = $f(A, B[:C])
      A_mul_B!(C, tmp*B[:U], B[:R])
    end
  end
end

for (f, g) in ((:Ac_mul_Bc, :A_mul_Bc), (:At_mul_Bt, :A_mul_Bt))
  f! = Symbol(f, "!")
  g! = Symbol(g, "!")
  @eval begin
    function $f!(C::StridedMatrix{T}, A::StridedMatrix{T}, B::CUR{T}) where T
      tmp = $f(A, B[:R])
      tmp = $g(tmp, B[:U])
      $g!(C, tmp, B[:C])
    end
  end
end

## HermCUR left-multiplication

A_mul_B!(C::StridedVecOrMat{T}, A::HermCUR{T}, B::StridedVecOrMat{T}) where {T} =
  A_mul_B!(C, A[:C], A[:U]*(A[:C]'*B))

A_mul_Bc!(C::StridedMatrix{T}, A::HermCUR{T}, B::StridedMatrix{T}) where {T} =
  A_mul_B!(C, A[:C], A[:U]*(A[:C]'*B'))
A_mul_Bt!(C::StridedMatrix{T}, A::HermCUR{T}, B::StridedMatrix{T}) where {T<:Real} =
  A_mul_B!(C, A[:C], A[:U]*(A[:C].'*B.'))
A_mul_Bt!!(
C::StridedMatrix{T}, A::HermCUR{T}, B::StridedMatrix{T}) where {T<:Complex} =
  A_mul_Bc!(C, A, conj!(B))  # overwrites B
function A_mul_Bt!(
    C::StridedMatrix{T}, A::HermCUR{T}, B::StridedMatrix{T}) where T<:Complex
  size(B, 1) <= A[:k] && return A_mul_Bt!!(C, A, copy(B))
  A_mul_B!(C, A[:C], A[:U]*((A[:C]')*B.'))
end

Ac_mul_B!(C::StridedVecOrMat{T}, A::HermCUR{T}, B::StridedVecOrMat{T}) where {T} =
  A_mul_B!(C, A, B)
function At_mul_B!(
    C::StridedVecOrMat{T}, A::HermCUR{T}, B::StridedVecOrMat{T}) where T
  tmp = A[:U].'*(A[:C].'*B)
  A_mul_B!(C, A[:C], conj!(tmp))
  conj!(C)
end

Ac_mul_Bc!(C::StridedMatrix{T}, A::HermCUR{T}, B::StridedMatrix{T}) where {T} =
  A_mul_Bc!(C, A, B)
function At_mul_Bt!(C::StridedMatrix{T}, A::HermCUR{T}, B::StridedMatrix{T}) where T
  tmp = A[:U].'*(A[:C].'*B.')
  A_mul_B!(C, A[:C], conj!(tmp))
  conj!(C)
end

## HermCUR right-multiplication

A_mul_B!(C::StridedMatrix{T}, A::StridedMatrix{T}, B::HermCUR{T}) where {T} =
  A_mul_Bc!(C, (A*B[:C])*B[:U], B[:C])

A_mul_Bc!(C::StridedMatrix{T}, A::StridedMatrix{T}, B::HermCUR{T}) where {T} =
  A_mul_B!(C, A, B)
function A_mul_Bt!!(C::StridedMatrix{T}, A::StridedMatrix{T}, B::HermCUR{T}) where T
  tmp = conj!(A)*B[:C]
  A_mul_Bt!(C, conj!(tmp)*B[:U].', B[:C])
end  # overwrites A
function A_mul_Bt!(C::StridedMatrix{T}, A::StridedMatrix{T}, B::HermCUR{T}) where T
  size(A, 1) <= B[:k] && return A_mul_Bt!!(C, copy(A), B)
  A_mul_Bt!(C, (A*conj(B[:C]))*B[:U].', B[:C])
end

for f in (:Ac_mul_B, :At_mul_B)
  f! = Symbol(f, "!")
  @eval begin
    function $f!(C::StridedMatrix{T}, A::StridedMatrix{T}, B::HermCUR{T}) where T
      tmp = $f(A, B[:C])
      A_mul_Bc!(C, tmp*B[:U], B[:C])
    end
  end
end

Ac_mul_Bc!(C::StridedMatrix{T}, A::StridedMatrix{T}, B::HermCUR{T}) where {T} =
  Ac_mul_B!(C, A, B)
At_mul_Bt!(C::StridedMatrix{T}, A::StridedMatrix{T}, B::HermCUR{T}) where {T} =
  A_mul_Bt!(C, conj!(A'*B[:C])*B[:U].', B[:C])

## SymCUR left-multiplication

A_mul_B!(C::StridedVecOrMat{T}, A::SymCUR{T}, B::StridedVecOrMat{T}) where {T} =
  A_mul_B!(C, A[:C], A[:U]*(A[:C].'*B))
A_mul_Bc!(C::StridedMatrix{T}, A::SymCUR{T}, B::StridedMatrix{T}) where {T<:Real} =
  A_mul_B!(C, A[:C], A[:U]*(A[:C]'*B'))
A_mul_Bc!!(C::StridedMatrix{T}, A::SymCUR{T}, B::StridedMatrix{T}) where {T<:Complex} =
  A_mul_Bt!(C, A, conj!(B))  # overwrites B
function A_mul_Bc!(
    C::StridedMatrix{T}, A::SymCUR{T}, B::StridedMatrix{T}) where T<:Complex
  size(B, 1) <= A[:k] && return A_mul_Bc!!(C, A, copy(B))
  A_mul_B!(C, A[:C], A[:U]*((A[:C].')*B'))
end
A_mul_Bt!(C::StridedMatrix{T}, A::SymCUR{T}, B::StridedMatrix{T}) where {T} =
  A_mul_B!(C, A[:C], A[:U]*(A[:C].'*B.'))

function Ac_mul_B!(
    C::StridedVecOrMat{T}, A::SymCUR{T}, B::StridedVecOrMat{T}) where T
  tmp = A[:U]'*(A[:C]'*B)
  A_mul_B!(C, A[:C], conj!(tmp))
  conj!(C)
end
At_mul_B!(C::StridedVecOrMat{T}, A::SymCUR{T}, B::StridedVecOrMat{T}) where {T} =
  A_mul_B!(C, A, B)

function Ac_mul_Bc!(C::StridedMatrix{T}, A::SymCUR{T}, B::StridedMatrix{T}) where T
  tmp = A[:U]'*(A[:C]'*B')
  A_mul_B!(C, A[:C], conj!(tmp))
  conj!(C)
end
At_mul_Bt!(C::StridedMatrix{T}, A::SymCUR{T}, B::StridedMatrix{T}) where {T} =
  A_mul_Bt!(C, A, B)

## SymCUR right-multiplication

A_mul_B!(C::StridedMatrix{T}, A::StridedMatrix{T}, B::SymCUR{T}) where {T} =
  A_mul_Bt!(C, (A*B[:C])*B[:U], B[:C])

function A_mul_Bc!!(C::StridedMatrix{T}, A::StridedMatrix{T}, B::SymCUR{T}) where T
  tmp = conj!(A)*B[:C]
  A_mul_Bt!(C, conj!(tmp)*B[:U].', B[:C])
end  # overwrites A
function A_mul_Bc!(C::StridedMatrix{T}, A::StridedMatrix{T}, B::SymCUR{T}) where T
  size(A, 1) <= B[:k] && return A_mul_Bc!!(C, copy(A), B)
  A_mul_Bc!(C, (A*conj(B[:C]))*B[:U]', B[:C])
end
A_mul_Bt!(C::StridedMatrix{T}, A::StridedMatrix{T}, B::SymCUR{T}) where {T} =
  A_mul_B!(C, A, B)

for f in (:Ac_mul_B, :At_mul_B)
  f! = Symbol(f, "!")
  @eval begin
    function $f!(C::StridedMatrix{T}, A::StridedMatrix{T}, B::SymCUR{T}) where T
      tmp = $f(A, B[:C])
      A_mul_Bt!(C, tmp*B[:U], B[:C])
    end
  end
end

Ac_mul_Bc!(C::StridedMatrix{T}, A::StridedMatrix{T}, B::SymCUR{T}) where {T} =
  A_mul_Bc!(C, conj!(A.'*B[:C])*B[:U]', B[:C])
At_mul_Bt!(C::StridedMatrix{T}, A::StridedMatrix{T}, B::SymCUR{T}) where {T} =
  At_mul_B!(C, A, B)

# standard operations

## left-multiplication

for (f, f!, i) in ((:*,        :A_mul_B!,  1),
                   (:Ac_mul_B, :Ac_mul_B!, 2),
                   (:At_mul_B, :At_mul_B!, 2))
  @eval begin
    function $f(A::AbstractCUR{TA}, B::StridedVector{TB}) where {TA,TB}
      T = promote_type(TA, TB)
      AT = convert(AbstractCUR{T}, A)
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
    function $f(A::AbstractCUR{TA}, B::StridedMatrix{TB}) where {TA,TB}
      T = promote_type(TA, TB)
      AT = convert(AbstractCUR{T}, A)
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
    function $f(A::StridedMatrix{TA}, B::AbstractCUR{TB}) where {TA,TB}
      T = promote_type(TA, TB)
      AT = (T == TA ? A : convert(Array{T}, A))
      BT = convert(AbstractCUR{T}, B)
      CT = Array{T}(size(A,$i), size(B,$j))
      $f!(CT, AT, BT)
    end
  end
end

# factorization routines

for sfx in ("", "!")
  f = Symbol("curfact", sfx)
  g = Symbol("curfact_none", sfx)
  h = Symbol("cur", sfx)
  @eval begin
    function $f(
        A::AbstractMatOrLinOp{T}, opts::LRAOptions=LRAOptions(T); args...) where T
      opts = copy(opts; args...)
      opts.pqrfact_retval = ""
      chkopts!(opts, A)
      opts.sketch == :none && return $g(A, opts)
      if ishermitian(A)
        F = sketchfact(:left, :n, A, opts)
        cols = F[:p][1:F[:k]]
        return HermCURPackedU(cols)
      elseif issymmetric(A)
        F = sketchfact(:left, :n, A, opts)
        cols = F[:p][1:F[:k]]
        return SymCURPackedU(cols)
      end
      m, n = size(A)
      if m >= n
        F = sketchfact(:left, :c, A, opts)
        rows = F[:p][1:F[:k]]
        F = sketchfact(:left, :n, A[rows,:], opts)
        cols = F[:p][1:F[:k]]
      else
        F = sketchfact(:left, :n, A, opts)
        cols = F[:p][1:F[:k]]
        F = sketchfact(:left, :c, A[:,cols], opts)
        rows = F[:p][1:F[:k]]
      end
      kr = length(rows)
      kc = length(cols)
      k = min(kr, kc)
      rows = k < kr ? rows[1:k] : rows
      cols = k < kc ? cols[1:k] : cols
      CURPackedU(rows, cols)
    end
    $f(A, args...; kwargs...) = $f(LinOp(A), args...; kwargs...)

    function $h(A, args...; kwargs...)
      U = $f(A, args...; kwargs...)
      U[:rows], U[:cols]
    end
  end
end

function curfact_none!(A::StridedMatrix, opts::LRAOptions)
  if ishermitian(A)
    F = pqrfact_backend!(A, opts)
    cols = F[:p][1:F[:k]]
    return HermCURPackedU(cols)
  elseif issymmetric(A)
    F = pqrfact_backend!(A, opts)
    cols = F[:p][1:F[:k]]
    return SymCURPackedU(cols)
  end
  curfact_none(A, opts)
end
function curfact_none(A::StridedMatrix, opts::LRAOptions)
  (ishermitian(A) || issymmetric(A)) && return curfact_none!(copy(A), opts)
  m, n = size(A)
  if m >= n
    F = pqrfact_backend!(A', opts)
    rows = F[:p][1:F[:k]]
    F = pqrfact_backend!(A[rows,:], opts)
    cols = F[:p][1:F[:k]]
  else
    F = pqrfact_backend!(copy(A), opts)
    cols = F[:p][1:F[:k]]
    F = pqrfact_backend!(A[:,cols]', opts)
    rows = F[:p][1:F[:k]]
  end
  kr = length(rows)
  kc = length(cols)
  k = min(kr, kc)
  rows = k < kr ? rows[1:k] : rows
  cols = k < kc ? cols[1:k] : cols
  return CURPackedU(rows, cols)
end
