#= src/psvd.jl

References:

  N. Halko, P.G. Martinsson, J.A. Tropp. Finding structure with randomness:
    Probabilistic algorithms for constructing approximate matrix
    decompositions. SIAM Rev. 53 (2): 217-288, 2011.
=#

type PartialSVD{T,Tr<:Real} <: Factorization{T}
  U::Matrix{T}
  S::Vector{Tr}
  Vt::Matrix{T}
end

conj!(A::PartialSVD) = PartialSVD(conj!(A.U), A.S, conj!(A.Vt))
conj(A::PartialSVD) = conj!(copy(A))

function convert{T}(::Type{PartialSVD{T}}, A::PartialSVD)
  Tr = eltype(real(one(T)))
  PartialSVD(
    convert(Array{T}, A.U), convert(Array{Tr}, A.S), convert(Array{T}, A.Vt))
end
convert(::Type{Array}, A::PartialSVD) = full(A)
convert{T}(::Type{Array{T}}, A::PartialSVD) = convert(Array{T}, full(A))

copy(A::PartialSVD) = PartialSVD(copy(A.U), copy(A.S), copy(A.Vt))

full(A::PartialSVD) = scale(A[:U], A[:S])*A[:Vt]

function getindex(A::PartialSVD, d::Symbol)
  if     d == :S   return A.S
  elseif d == :U   return A.U
  elseif d == :Vt  return A.Vt
  elseif d == :k   return length(A.S)
  else             throw(KeyError(d))
  end
end

ishermitian(A::PartialSVD) = false
issym(A::PartialSVD) = false

isreal{T}(A::PartialSVD{T}) = T <: Real

ndims(A::PartialSVD) = 2

size(A::PartialSVD) = (size(A.U,1), size(A.Vt,2))
size(A::PartialSVD, dim::Integer) =
  dim == 1 ? size(A.U,1) : (dim == 2 ? size(A.Vt,2) : 1)

# BLAS/LAPACK multiplication routines

## left-multiplication

A_mul_B!{T}(y::StridedVector{T}, A::PartialSVD{T}, x::StridedVector{T}) =
  A_mul_B!(y, A[:U], scalevec!(A[:S], A[:Vt]*x))
A_mul_B!{T}(C::StridedMatrix{T}, A::PartialSVD{T}, B::StridedMatrix{T}) =
  A_mul_B!(C, A[:U], scale!(A[:S], A[:Vt]*B))

for f in (:A_mul_Bc, :A_mul_Bt)
  f! = symbol(f, "!")
  @eval begin
    function $f!{T}(C::StridedMatrix{T}, A::PartialSVD{T}, B::StridedMatrix{T})
      tmp = $f(A[:Vt], B)
      scale!(A[:S], tmp)
      A_mul_B!(C, A[:U], tmp)
    end
  end
end

for f in (:Ac_mul_B, :At_mul_B)
  f! = symbol(f, "!")
  @eval begin
    function $f!{T}(y::StridedVector{T}, A::PartialSVD{T}, x::StridedVector{T})
      tmp = $f(A[:U], x)
      scalevec!(A[:S], tmp)
      $f!(y, A[:Vt], tmp)
    end
    function $f!{T}(C::StridedMatrix{T}, A::PartialSVD{T}, B::StridedMatrix{T})
      tmp = $f(A[:U], B)
      scale!(A[:S], tmp)
      $f!(C, A[:Vt], tmp)
    end
  end
end

for (f, g!) in ((:Ac_mul_Bc, :Ac_mul_B!), (:At_mul_Bt, :At_mul_B!))
  f! = symbol(f, "!")
  @eval begin
    function $f!{T}(C::StridedMatrix{T}, A::PartialSVD{T}, B::StridedMatrix{T})
      tmp = $f(A[:U], B)
      scale!(A[:S], tmp)
      $g!(C, A[:Vt], tmp)
    end
  end
end

## right-multiplication

A_mul_B!{T}(C::StridedMatrix{T}, A::StridedMatrix{T}, B::PartialSVD{T}) =
  A_mul_B!(C, scale!(A*B[:U], B[:S]), B[:Vt])

for f in (:A_mul_Bc, :A_mul_Bt)
  f! = symbol(f, "!")
  @eval begin
    function $f!{T}(C::StridedMatrix{T}, A::StridedMatrix{T}, B::PartialSVD{T})
      tmp = $f(A, B[:Vt])
      scale!(tmp, B[:S])
      $f!(C, tmp, B[:U])
    end
  end
end

for f in (:Ac_mul_B, :At_mul_B)
  f! = symbol(f, "!")
  @eval begin
    function $f!{T}(C::StridedMatrix{T}, A::StridedMatrix{T}, B::PartialSVD{T})
      tmp = $f(A, B[:U])
      scale!(tmp, B[:S])
      A_mul_B!(C, tmp, B[:Vt])
    end
  end
end

for (f, g!) in ((:Ac_mul_Bc, :A_mul_Bc!), (:At_mul_Bt, :A_mul_Bt!))
  f! = symbol(f, "!")
  @eval begin
    function $f!{T}(C::StridedMatrix{T}, A::StridedMatrix{T}, B::PartialSVD{T})
      tmp = $f(A, B[:Vt])
      scale!(tmp, B[:S])
      $g!(C, tmp, B[:U])
    end
  end
end

# standard operations

## left-multiplication

for (f, f!, i) in ((:*,        :A_mul_B!,  1),
                   (:Ac_mul_B, :Ac_mul_B!, 2),
                   (:At_mul_B, :At_mul_B!, 2))
  @eval begin
    function $f{TA,TB}(A::PartialSVD{TA}, B::StridedVector{TB})
      T = promote_type(TA, TB)
      AT = convert(PartialSVD{T}, A)
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
    function $f{TA,TB}(A::PartialSVD{TA}, B::StridedMatrix{TB})
      T = promote_type(TA, TB)
      AT = convert(PartialSVD{T}, A)
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
    function $f{TA,TB}(A::StridedMatrix{TA}, B::PartialSVD{TB})
      T = promote_type(TA, TB)
      AT = (T == TA ? A : convert(Array{T}, A))
      BT = convert(PartialSVD{T}, B)
      CT = Array(T, size(A,$i), size(B,$j))
      $f!(CT, AT, BT)
    end
  end
end

# factorization routines

function psvdfact(A::AbstractMatOrLinOp, opts::LRAOptions)
  chkopts(opts)
  if ishermitian(A)
    Q = opts.sketch == :none ? pqrfact(A, opts)[:Q] : rrange(:n, A, opts)
    F = svdfact!(Q'*A*Q)
    U  = Q*F.U
    Vt = U'
  else
    m, n = size(A)
    if m >= n
      Q = opts.sketch == :none ? pqrfact(A, opts)[:Q] : rrange(:n, A, opts)
      F = svdfact!(Q'*A)
      U  = Q*F.U
      Vt = F.Vt
    else
      Q = opts.sketch == :none ? pqrfact!(A', opts)[:Q] : rrange(:c, A, opts)
      F = svdfact!(A*Q)
      U  = F.U
      Vt = F.Vt*Q'
    end
  end
  PartialSVD(U, F.S, Vt)
end
function psvdfact(A::AbstractMatOrLinOp, rank_or_rtol::Real)
  opts = (rank_or_rtol < 1 ? LRAOptions(rtol=rank_or_rtol)
                           : LRAOptions(rank=rank_or_rtol))
  psvdfact(A, opts)
end
psvdfact{T}(A::AbstractMatOrLinOp{T}) = psvdfact(A, default_rtol(T))
psvdfact(A, args...) = psvdfact(LinOp(A), args...)

function psvd(A::AbstractMatOrLinOp, args...)
  F = psvdfact(A, args...)
  F.U, F.S, F.Vt'
end

function psvdvals(A::AbstractMatOrLinOp, opts::LRAOptions)
  chkopts(opts)
  if ishermitian(A)
    Q = opts.sketch == :none ? pqrfact(A, opts)[:Q] : rrange(:n, A, opts)
    return svdvals!(Q'*A*Q)
  end
  m, n = size(A)
  if m >= n
    Q = opts.sketch == :none ? pqrfact(A, opts)[:Q] : rrange(:n, A, opts)
    return svdvals!(Q'*A)
  else
    Q = opts.sketch == :none ? pqrfact!(A', opts)[:Q] : rrange(:c, A, opts)
    return svdvals!(A*Q)
  end
end
function psvdvals(A::AbstractMatOrLinOp, rank_or_rtol::Real)
  opts = (rank_or_rtol < 1 ? LRAOptions(rtol=rank_or_rtol)
                           : LRAOptions(rank=rank_or_rtol))
  psvdvals(A, opts)
end
psvdvals{T}(A::AbstractMatOrLinOp{T}) = psvdvals(A, default_rtol(T))
psvdvals(A, args...) = psvdvals(LinOp(A), args...)