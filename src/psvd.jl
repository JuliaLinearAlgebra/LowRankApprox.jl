#= src/psvd.jl
=#

mutable struct PartialSVD{T,Tr<:Real} <: Factorization{T}
  U::Matrix{T}
  S::Vector{Tr}
  Vt::Matrix{T}
end

PartialSVD(U::AbstractMatrix{T}, S::AbstractVector, Vt::AbstractMatrix{T}) where T =
  PartialSVD(Matrix(U), Vector(S), Matrix(Vt))

conj!(A::PartialSVD) = PartialSVD(conj!(A.U), A.S, conj!(A.Vt))
conj(A::PartialSVD) = PartialSVD(conj(A.U), A.S, conj(A.Vt))

function convert(::Type{PartialSVD{T}}, A::PartialSVD) where T
  Tr = real(T)
  PartialSVD(
    convert(Array{T}, A.U), convert(Array{Tr}, A.S), convert(Array{T}, A.Vt))
end
convert(::Type{Factorization{T}}, A::PartialSVD) where {T} = convert(PartialSVD{T}, A)
convert(::Type{Factorization{T}}, A::PartialSVD{T,Tr}) where {T,Tr<:Real} = convert(PartialSVD{T}, A)
convert(::Type{Array}, A::PartialSVD) = full(A)
convert(::Type{Array{T}}, A::PartialSVD) where {T} = convert(Array{T}, full(A))

copy(A::PartialSVD) = PartialSVD(copy(A.U), copy(A.S), copy(A.Vt))

adjoint!(A::PartialSVD) = PartialSVD(A.Vt', A.S, A.U')
adjoint(A::PartialSVD) = PartialSVD(A.Vt', copy(A.S), A.U')
transpose!(A::PartialSVD) = PartialSVD(transpose(A.Vt), A.S, transpose(A.U))
transpose(A::PartialSVD) = PartialSVD(transpose(A.Vt), copy(A.S), transpose(A.U))

full(A::PartialSVD) = (A[:U]*Diagonal(A[:S]))*A[:Vt]

function getindex(A::PartialSVD, d::Symbol)
  if     d == :S   return A.S
  elseif d == :U   return A.U
  elseif d == :V   return A.Vt'
  elseif d == :Vt  return A.Vt
  elseif d == :k   return length(A.S)
  else             throw(KeyError(d))
  end
end

ishermitian(::PartialSVD) = false
issym(::PartialSVD) = false

isreal(::PartialSVD{T}) where {T} = T <: Real

ndims(::PartialSVD) = 2

size(A::PartialSVD) = (size(A.U,1), size(A.Vt,2))
size(A::PartialSVD, dim::Integer) =
  dim == 1 ? size(A.U,1) : (dim == 2 ? size(A.Vt,2) : 1)

# BLAS/LAPACK multiplication/division routines

## left-multiplication

mul!(y::StridedVector{T}, A::PartialSVD{T}, x::StridedVector{T}) where {T} =
  mul!(y, A[:U], scalevec!(A[:S], A[:Vt]*x))
mul!(C::StridedMatrix{T}, A::PartialSVD{T}, B::StridedMatrix{T}) where {T} =
  mul!(C, A[:U], scale!(A[:S], A[:Vt]*B))

for f in (:A_mul_Bc, :A_mul_Bt)
  f! = Symbol(f, "!")
  @eval begin
    function $f!(C::StridedMatrix{T}, A::PartialSVD{T}, B::StridedMatrix{T}) where T
      tmp = $f(A[:Vt], B)
      scale!(A[:S], tmp)
      mul!(C, A[:U], tmp)
    end
  end
end

for f in (:Ac_mul_B, :At_mul_B)
  f! = Symbol(f, "!")
  @eval begin
    function $f!(
        y::StridedVector{T}, A::PartialSVD{T}, x::StridedVector{T}) where T
      tmp = $f(A[:U], x)
      scalevec!(A[:S], tmp)
      $f!(y, A[:Vt], tmp)
    end
    function $f!(
        C::StridedMatrix{T}, A::PartialSVD{T}, B::StridedMatrix{T}) where T
      tmp = $f(A[:U], B)
      scale!(A[:S], tmp)
      $f!(C, A[:Vt], tmp)
    end
  end
end

for (f, g!) in ((:Ac_mul_Bc, :Ac_mul_B!), (:At_mul_Bt, :At_mul_B!))
  f! = Symbol(f, "!")
  @eval begin
    function $f!(C::StridedMatrix{T}, A::PartialSVD{T}, B::StridedMatrix{T}) where T
      tmp = $f(A[:U], B)
      scale!(A[:S], tmp)
      $g!(C, A[:Vt], tmp)
    end
  end
end

## right-multiplication

mul!(C::StridedMatrix{T}, A::StridedMatrix{T}, B::PartialSVD{T}) where {T} =
  mul!(C, scale!(A*B[:U], B[:S]), B[:Vt])

for f in (:A_mul_Bc, :A_mul_Bt)
  f! = Symbol(f, "!")
  @eval begin
    function $f!(C::StridedMatrix{T}, A::StridedMatrix{T}, B::PartialSVD{T}) where T
      tmp = $f(A, B[:Vt])
      scale!(tmp, B[:S])
      $f!(C, tmp, B[:U])
    end
  end
end

for f in (:Ac_mul_B, :At_mul_B)
  f! = Symbol(f, "!")
  @eval begin
    function $f!(C::StridedMatrix{T}, A::StridedMatrix{T}, B::PartialSVD{T}) where T
      tmp = $f(A, B[:U])
      scale!(tmp, B[:S])
      mul!(C, tmp, B[:Vt])
    end
  end
end

for (f, g!) in ((:Ac_mul_Bc, :A_mul_Bc!), (:At_mul_Bt, :A_mul_Bt!))
  f! = Symbol(f, "!")
  @eval begin
    function $f!(C::StridedMatrix{T}, A::StridedMatrix{T}, B::PartialSVD{T}) where T
      tmp = $f(A, B[:Vt])
      scale!(tmp, B[:S])
      $g!(C, tmp, B[:U])
    end
  end
end

## left-division (pseudoinverse left-multiplication)
A_ldiv_B!(y::StridedVector{T}, A::PartialSVD{T}, x::StridedVector{T}) where {T} =
  Ac_mul_B!(y, A[:Vt], iscalevec!(A[:S], A[:U]'*x))
A_ldiv_B!(C::StridedMatrix{T}, A::PartialSVD{T}, B::StridedMatrix{T}) where {T} =
  Ac_mul_B!(C, A[:Vt], iscale!(A[:S], A[:U]'*B))

# standard operations

## left-multiplication

for (f, f!, i) in ((:*,        :mul!,  1),
                   (:Ac_mul_B, :Ac_mul_B!, 2),
                   (:At_mul_B, :At_mul_B!, 2))
  @eval begin
    function $f(A::PartialSVD{TA}, B::StridedVector{TB}) where {TA,TB}
      T = promote_type(TA, TB)
      AT = convert(PartialSVD{T}, A)
      BT = (T == TB ? B : convert(Array{T}, B))
      CT = Array{T}(undef, size(A,$i))
      $f!(CT, AT, BT)
    end
  end
end

for (f, f!, i, j) in ((:*,         :mul!,   1, 2),
                      (:A_mul_Bc,  :A_mul_Bc!,  1, 1),
                      (:A_mul_Bt,  :A_mul_Bt!,  1, 1),
                      (:Ac_mul_B,  :Ac_mul_B!,  2, 2),
                      (:Ac_mul_Bc, :Ac_mul_Bc!, 2, 1),
                      (:At_mul_B,  :At_mul_B!,  2, 2),
                      (:At_mul_Bt, :At_mul_Bt!, 2, 1))
  @eval begin
    function $f(A::PartialSVD{TA}, B::StridedMatrix{TB}) where {TA,TB}
      T = promote_type(TA, TB)
      AT = convert(PartialSVD{T}, A)
      BT = (T == TB ? B : convert(Array{T}, B))
      CT = Array{T}(undef, size(A,$i), size(B,$j))
      $f!(CT, AT, BT)
    end
  end
end

## right-multiplication
for (f, f!, i, j) in ((:*,         :mul!,   1, 2),
                      (:A_mul_Bc,  :A_mul_Bc!,  1, 1),
                      (:A_mul_Bt,  :A_mul_Bt!,  1, 1),
                      (:Ac_mul_B,  :Ac_mul_B!,  2, 2),
                      (:Ac_mul_Bc, :Ac_mul_Bc!, 2, 1),
                      (:At_mul_B,  :At_mul_B!,  2, 2),
                      (:At_mul_Bt, :At_mul_Bt!, 2, 1))
  @eval begin
    function $f(A::StridedMatrix{TA}, B::PartialSVD{TB}) where {TA,TB}
      T = promote_type(TA, TB)
      AT = (T == TA ? A : convert(Array{T}, A))
      BT = convert(PartialSVD{T}, B)
      CT = Array{T}(undef, size(A,$i), size(B,$j))
      $f!(CT, AT, BT)
    end
  end
end

## left-division
function \(A::PartialSVD{TA}, B::StridedVector{TB}) where {TA,TB}
  T = promote_type(TA, TB)
  AT = convert(PartialSVD{T}, A)
  BT = (T == TB ? B : convert(Array{T}, B))
  CT = Array{T}(undef, size(A,2))
  A_ldiv_B!(CT, AT, BT)
end
function \(A::PartialSVD{TA}, B::StridedMatrix{TB}) where {TA,TB}
  T = promote_type(TA, TB)
  AT = convert(PartialSVD{T}, A)
  BT = (T == TB ? B : convert(Array{T}, B))
  CT = Array{T}(undef, size(A,2), size(B,2))
  A_ldiv_B!(CT, AT, BT)
end

# factorization routines

function psvdfact(
    A::AbstractMatOrLinOp{T}, opts::LRAOptions=LRAOptions(T); args...) where T
  opts = isempty(args) ? opts : copy(opts; args...)
  m, n = size(A)
  if m >= n
    V = idfact(:n, A, opts)
    F = qrfact!(getcols(:n, A, V[:sk]))
    Q = F[:Q]
    F = svdfact!(F[:R]*V)
    k = psvdrank(F[:S], opts)
    if k < V[:k]
      U  = Q*view(F.U,:,1:k)
      S  = F.S[1:k]
      Vt = F.Vt[1:k,:]
    else
      U  = Q*F.U
      S  = F.S
      Vt = F.Vt
    end
  else
    V = idfact(:c, A, opts)
    F = qrfact!(getcols(:c, A, V[:sk]))
    Q = F[:Q]
    F = svdfact!(V'*F[:R]')
    k = psvdrank(F[:S], opts)
    if k < V[:k]
      U  = F.U[:,1:k]
      S  = F.S[1:k]
      Vt = view(F.Vt,1:k,:)*Q'
    else
      U  = F.U
      S  = F.S
      Vt = F.Vt*Q'
    end
  end
  PartialSVD(U, S, Vt)
end

function psvdvals(
    A::AbstractMatOrLinOp{T}, opts::LRAOptions=LRAOptions(T); args...) where T
  opts = isempty(args) ? opts : copy(opts; args...)
  m, n = size(A)
  if m >= n
    V = idfact(:n, A, opts)
    F = qrfact!(getcols(:n, A, V[:sk]))
    s = svdvals!(F[:R]*V)
  else
    V = idfact(:c, A, opts)
    F = qrfact!(getcols(:c, A, V[:sk]))
    s = svdvals!(V'*F[:R]')
  end
  k = psvdrank(s, opts)
  k < V[:k] && return s[1:k]
  s
end

for f in (:psvdfact, :psvdvals)
  @eval $f(A, args...; kwargs...) = $f(LinOp(A), args...; kwargs...)
end

function psvd(A, args...; kwargs...)
  F = psvdfact(A, args...; kwargs...)
  F.U, F.S, F.Vt'
end

function psvdrank(s::Vector{T}, opts::LRAOptions) where T<:Real
  k = length(s)
  ptol = max(opts.atol, opts.rtol*s[1])
  @inbounds for i = 2:k
    s[i] <= ptol && return i - 1
  end
  k
end
