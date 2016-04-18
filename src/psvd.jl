#= src/psvd.jl
=#

type PartialSVD{T,Tr<:Real} <: Factorization{T}
  U::Matrix{T}
  S::Vector{Tr}
  Vt::Matrix{T}
end

conj!(A::PartialSVD) = PartialSVD(conj!(A.U), A.S, conj!(A.Vt))
conj(A::PartialSVD) = PartialSVD(conj(A.U), A.S, conj(A.Vt))

function convert{T}(::Type{PartialSVD{T}}, A::PartialSVD)
  Tr = real(T)
  PartialSVD(
    convert(Array{T}, A.U), convert(Array{Tr}, A.S), convert(Array{T}, A.Vt))
end
convert{T}(::Type{Factorization{T}}, A::PartialSVD) = convert(PartialSVD{T}, A)
convert(::Type{Array}, A::PartialSVD) = full(A)
convert{T}(::Type{Array{T}}, A::PartialSVD) = convert(Array{T}, full(A))

copy(A::PartialSVD) = PartialSVD(copy(A.U), copy(A.S), copy(A.Vt))

ctranspose!(A::PartialSVD) = PartialSVD(A.Vt', A.S, A.U')
ctranspose(A::PartialSVD) = PartialSVD(A.Vt', copy(A.S), A.U')
transpose!(A::PartialSVD) = PartialSVD(A.Vt.', A.S, A.U.')
transpose(A::PartialSVD) = PartialSVD(A.Vt.', copy(A.S), A.U.')

full(A::PartialSVD) = scale(A[:U], A[:S])*A[:Vt]

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

isreal{T}(::PartialSVD{T}) = T <: Real

ndims(::PartialSVD) = 2

size(A::PartialSVD) = (size(A.U,1), size(A.Vt,2))
size(A::PartialSVD, dim::Integer) =
  dim == 1 ? size(A.U,1) : (dim == 2 ? size(A.Vt,2) : 1)

# BLAS/LAPACK multiplication/division routines

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
    function $f!{T}(
        y::StridedVector{T}, A::PartialSVD{T}, x::StridedVector{T})
      tmp = $f(A[:U], x)
      scalevec!(A[:S], tmp)
      $f!(y, A[:Vt], tmp)
    end
    function $f!{T}(
        C::StridedMatrix{T}, A::PartialSVD{T}, B::StridedMatrix{T})
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

## left-division (pseudoinverse left-multiplication)
A_ldiv_B!{T}(y::StridedVector{T}, A::PartialSVD{T}, x::StridedVector{T}) =
  Ac_mul_B!(y, A[:Vt], iscalevec!(A[:S], A[:U]'*x))
A_ldiv_B!{T}(C::StridedMatrix{T}, A::PartialSVD{T}, B::StridedMatrix{T}) =
  Ac_mul_B!(C, A[:Vt], iscale!(A[:S], A[:U]'*B))

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

## left-division
function \{TA,TB}(A::PartialSVD{TA}, B::StridedVector{TB})
  T = promote_type(TA, TB)
  AT = convert(PartialSVD{T}, A)
  BT = (T == TB ? B : convert(Array{T}, B))
  CT = Array(T, size(A,2))
  A_ldiv_B!(CT, AT, BT)
end
function \{TA,TB}(A::PartialSVD{TA}, B::StridedMatrix{TB})
  T = promote_type(TA, TB)
  AT = convert(PartialSVD{T}, A)
  BT = (T == TB ? B : convert(Array{T}, B))
  CT = Array(T, size(A,2), size(B,2))
  A_ldiv_B!(CT, AT, BT)
end

# factorization routines

function psvdfact{T}(
    A::AbstractMatOrLinOp{T}, opts::LRAOptions=LRAOptions(T); args...)
  opts = isempty(args) ? opts : copy(opts; args...)
  m, n = size(A)
  if m >= n
    V = idfact(:n, A, opts)
    F = qrfact!(getcols(:n, A, V[:sk]))
    Q = F[:Q]
    F = svdfact!(F[:R]*V)
    k = psvdrank(F[:S], opts)
    if k < V[:k]
      U  = Q*sub(F.U,:,1:k)
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
      Vt = sub(F.Vt,1:k,:)*Q'
    else
      U  = F.U
      S  = F.S
      Vt = F.Vt*Q'
    end
  end
  PartialSVD(U, S, Vt)
end

function psvdvals{T}(
    A::AbstractMatOrLinOp{T}, opts::LRAOptions=LRAOptions(T); args...)
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

function psvdrank{T<:Real}(s::Vector{T}, opts::LRAOptions)
  k = length(s)
  ptol = max(opts.atol, opts.rtol*s[1])
  @inbounds for i = 2:k
    s[i] <= ptol && return i - 1
  end
  k
end