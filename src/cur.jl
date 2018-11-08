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
  U, σ, V = svd!(C[rows,:])
  U2 = PartialSVD(Matrix(V), 1 ./ σ, Matrix(U'))
  CUR(rows, cols, C, U2, R)
end
CUR(A::AbstractMatOrLinOp, U::HermCURPackedU) = HermCUR(A, U)
CUR(A::AbstractMatOrLinOp, U::SymCURPackedU) = SymCUR(A, U)
CUR(A::AbstractMatOrLinOp, rows, cols) = CUR(A, CURPackedU(rows, cols))
CUR(A, args...) = CUR(LinOp(A), args...)


function HermCUR(A::AbstractMatOrLinOp, U::HermCURPackedU)
    cols = U[:cols]
    C = A[:,cols]
    F = eigen!(Hermitian(C[cols,:]))
    U = PartialHermEigen(1 ./ F.values, F.vectors)
    HermCUR(cols, C, U)
end

HermCUR(A::AbstractMatOrLinOp, cols) = HermCUR(A, HermCURPackedU(cols))
HermCUR(A, args...) = HermCUR(LinOp(A), args...)

function SymCUR(A::AbstractMatOrLinOp, U::SymCURPackedU)
  cols = U[:cols]
  C = A[:,cols]
  U, σ, V = svd!(C[cols,:])
  U2 = PartialSVD(V, 1 ./ σ, U')
  SymCUR(cols, C, U2)
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
convert(::Type{Array}, A::AbstractCUR) = Matrix(A)
convert(::Type{Array{T}}, A::AbstractCUR) where {T} = convert(Array{T}, Matrix(A))
convert(::Type{Matrix{T}}, A::AbstractCUR) where {T} = convert(Array{T}, Matrix(A))
Array(A::AbstractCUR) = convert(Array, A)
Matrix(A::AbstractCUR) = convert(Array, A)
Array{T}(A::AbstractCUR) where T = convert(Array{T}, A)
Matrix{T}(A::AbstractCUR) where T = convert(Array{T}, A)

copy(A::CUR) = CUR(copy(A.rows), copy(A.cols), copy(A.C), copy(A.U), copy(A.R))
copy(A::HermCUR) = HermCUR(copy(A.cols), copy(A.C), copy(A.U))
copy(A::SymCUR) = SymCUR(copy(A.cols), copy(A.C), copy(A.U))

Matrix(A::CUR{T}) where {T} = A[:C]*(A[:U]*A[:R])
Matrix(A::HermCUR{T}) where {T} = A[:C]*(A[:U]*A[:C]')
Matrix(A::SymCUR{T}) where {T} = A[:C]*(A[:U]*transpose(A[:C]))


  adjoint(A::AbstractCUR) = Adjoint(A)
  transpose(A::AbstractCUR) = Transpose(A)


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
  elseif d == :R              return transpose(A.C)
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

  mul!(C::AbstractVector{T}, A::CUR{T}, B::AbstractVector{T}) where {T} =
    mul!(C, A[:C], A[:U]*(A[:R]*B))
  mul!(C::AbstractMatrix{T}, A::CUR{T}, B::AbstractMatrix{T}) where {T} =
    mul!(C, A[:C], A[:U]*(A[:R]*B))

  ## CUR right-multiplication

  mul!(C::AbstractMatrix{T}, A::AbstractMatrix{T}, B::CUR{T}) where {T} =
    mul!(C, (A*B[:C])*B[:U], B[:R])

  ## HermCUR left-multiplication

  mul!(C::AbstractVector{T}, A::HermCUR{T}, B::AbstractVector{T}) where {T} =
    mul!(C, A[:C], A[:U]*(A[:C]'*B))
  mul!(C::AbstractMatrix{T}, A::HermCUR{T}, B::AbstractMatrix{T}) where {T} =
    mul!(C, A[:C], A[:U]*(A[:C]'*B))

  for Adj in (:Adjoint, :Transpose)
    @eval begin
      ## CUR left-multiplication
      function mul!(C::AbstractMatrix{T}, A::CUR{T}, Bc::$Adj{T,<:AbstractMatrix{T}}) where T
        tmp = A[:R]*Bc
        mul!(C, A[:C], A[:U]*tmp)
      end

      function mul!(C::AbstractVector{T}, Ac::$Adj{T,CUR{T}}, B::AbstractVector{T}) where T
        A = parent(Ac)
        tmp = $Adj(A[:C]) * B
        tmp = $Adj(A[:U]) * tmp
        mul!(C, $Adj(A[:R]), tmp)
      end
      function mul!(C::AbstractMatrix{T}, Ac::$Adj{T,CUR{T}}, B::AbstractMatrix{T}) where T
        A = parent(Ac)
        tmp = $Adj(A[:C]) * B
        tmp = $Adj(A[:U]) * tmp
        mul!(C, $Adj(A[:R]), tmp)
      end
      function mul!(C::AbstractMatrix{T}, Ac::$Adj{T,CUR{T}}, Bc::$Adj{T,<:AbstractMatrix{T}}) where T
        A = parent(Ac)
        tmp = $Adj(A[:C]) * Bc
        tmp = $Adj(A[:U]) * tmp
        mul!(C, $Adj(A[:R]), tmp)
      end
        ## CUR right-multiplication
      function mul!(C::AbstractMatrix{T}, A::AbstractMatrix{T}, Bc::$Adj{T,CUR{T}}) where T
        B = parent(Bc)
        tmp = A * $Adj(B[:R])
        tmp = tmp * $Adj(B[:U])
        mul!(C, tmp, $Adj(B[:C]))
      end
      function mul!(C::AbstractMatrix{T}, Ac::$Adj{T,<:AbstractMatrix{T}}, B::CUR{T}) where T
        tmp = Ac * B[:C]
        mul!(C, tmp*B[:U], B[:R])
      end

      function mul!(C::AbstractMatrix{T}, Ac::$Adj{T,<:AbstractMatrix{T}}, Bc::$Adj{T,CUR{T}}) where T
        B = parent(Bc)
        tmp = Ac * $Adj(B[:R])
        tmp = tmp * $Adj(B[:U])
        mul!(C, tmp, $Adj(B[:C]))
      end
    end
  end

  ## HermCUR left-multiplication


  mul!(C::AbstractMatrix{T}, A::HermCUR{T}, Bc::Adjoint{T,<:AbstractMatrix{T}}) where {T} =
    mul!(C, A[:C], A[:U]*(A[:C]'*Bc))
  mul!(C::AbstractMatrix{T}, A::HermCUR{T}, Bt::Transpose{T,<:AbstractMatrix{T}}) where {T<:Real} =
    mul!(C, A[:C], A[:U]*(transpose(A[:C])*Bt))
  mul!!(C::AbstractMatrix{T}, A::HermCUR{T}, Bt::Transpose{T,<:AbstractMatrix{T}}) where {T<:Complex} =
    mul!(C, A, conj!(parent(Bt))')  # overwrites B
  function mul!(C::AbstractMatrix{T}, A::HermCUR{T}, Bt::Transpose{T,<:AbstractMatrix{T}}) where T<:Complex
    B = parent(Bt)
    size(B, 1) <= A[:k] && return mul!!(C, A, transpose(copy(B)))
    mul!(C, A[:C], A[:U]*((A[:C]')*Bt))
  end

  mul!(C::AbstractVector{T}, Ac::Adjoint{T,HermCUR{T}}, B::AbstractVector{T}) where {T} =
    mul!(C, parent(Ac), B)
  mul!(C::AbstractMatrix{T}, Ac::Adjoint{T,HermCUR{T}}, B::AbstractMatrix{T}) where {T} =
    mul!(C, parent(Ac), B)
  function mul!(C::AbstractVector{T}, At::Transpose{T,HermCUR{T}}, B::AbstractVector{T}) where T
    A = parent(At)
    tmp = transpose(A[:U])*(transpose(A[:C])*B)
    mul!(C, A[:C], conj!(tmp))
    conj!(C)
  end
  function mul!(C::AbstractMatrix{T}, At::Transpose{T,HermCUR{T}}, B::AbstractMatrix{T}) where T
    A = parent(At)
    tmp = transpose(A[:U])*(transpose(A[:C])*B)
    mul!(C, A[:C], conj!(tmp))
    conj!(C)
  end

  mul!(C::AbstractMatrix{T}, At::Adjoint{T,HermCUR{T}}, Bt::Adjoint{T,<:AbstractMatrix{T}}) where {T} =
    mul!(C, parent(At), Bt)
  function mul!(C::AbstractMatrix{T}, At::Transpose{T,HermCUR{T}}, Bt::Transpose{T,<:AbstractMatrix{T}}) where T
    A = parent(At)
    B = parent(Bt)
    tmp = transpose(A[:U])*(transpose(A[:C])*Bt)
    mul!(C, A[:C], conj!(tmp))
    conj!(C)
  end

  ## HermCUR right-multiplication

  mul!(C::AbstractMatrix{T}, A::AbstractMatrix{T}, B::HermCUR{T}) where {T} =
    mul!(C, (A*B[:C])*B[:U], B[:C]')

  mul!(C::AbstractMatrix{T}, A::AbstractMatrix{T}, Bt::Adjoint{T,HermCUR{T}}) where {T} =
    mul!(C, A, parent(Bt))
  function mul!!(C::AbstractMatrix{T}, A::AbstractMatrix{T}, Bt::Transpose{T,HermCUR{T}}) where T
    B = parent(Bt)
    tmp = conj!(A)*B[:C]
    mul!(C, conj!(tmp)*transpose(B[:U]), transpose(B[:C]))
  end  # overwrites A
  function mul!(C::AbstractMatrix{T}, A::AbstractMatrix{T}, Bt::Transpose{T,HermCUR{T}}) where T
    B = parent(Bt)
    size(A, 1) <= B[:k] && return mul!!(C, copy(A), Bt)
    mul!(C, (A*conj(B[:C]))*transpose(B[:U]), transpose(B[:C]))
  end

  for Adj in (:Transpose, :Adjoint)
    @eval begin
      function mul!(C::AbstractMatrix{T}, Ac::$Adj{T,<:AbstractMatrix{T}}, B::HermCUR{T}) where T
        tmp = Ac * B[:C]
        mul!(C, tmp*B[:U], B[:C]')
      end
    end
  end

  mul!(C::AbstractMatrix{T}, Ac::Adjoint{T,<:AbstractMatrix{T}}, Bc::Adjoint{T,HermCUR{T}}) where {T} =
    mul!(C, Ac, parent(Bc))
  function mul!(C::AbstractMatrix{T}, At::Transpose{T,<:AbstractMatrix{T}}, Bt::Transpose{T,HermCUR{T}}) where {T}
    B = parent(Bt)
    A = parent(At)
    mul!(C, conj!(A'*B[:C])*transpose(B[:U]), transpose(B[:C]))
  end

  ## SymCUR left-multiplication

  mul!(C::AbstractVector{T}, A::SymCUR{T}, B::AbstractVector{T}) where {T} =
    mul!(C, A[:C], A[:U]*(transpose(A[:C])*B))
  mul!(C::AbstractMatrix{T}, A::SymCUR{T}, B::AbstractMatrix{T}) where {T} =
    mul!(C, A[:C], A[:U]*(transpose(A[:C])*B))
  mul!(C::AbstractMatrix{T}, A::SymCUR{T}, Bc::Adjoint{T,<:AbstractMatrix{T}}) where {T<:Real} =
    mul!(C, A[:C], A[:U]*(A[:C]'*Bc))
  mul!!(C::AbstractMatrix{T}, A::SymCUR{T}, Bc::Adjoint{T,<:AbstractMatrix{T}}) where {T<:Complex} =
    mul!(C, A, transpose(conj!(parent(Bc))))  # overwrites B
  function mul!(C::AbstractMatrix{T}, A::SymCUR{T}, Bc::Adjoint{T,<:AbstractMatrix{T}}) where T<:Complex
    B = parent(Bc)
    size(B, 1) <= A[:k] && return mul!!(C, A, copy(B)')
    mul!(C, A[:C], A[:U]*(transpose(A[:C])*B'))
  end
  mul!(C::AbstractMatrix{T}, A::SymCUR{T}, Bt::Transpose{T,<:AbstractMatrix{T}}) where {T} =
    mul!(C, A[:C], A[:U]*(transpose(A[:C])*Bt))

  function mul!(C::AbstractVector{T}, Ac::Adjoint{T,SymCUR{T}}, B::AbstractVector{T}) where T
    A = parent(Ac)
    tmp = A[:U]'*(A[:C]'*B)
    mul!(C, A[:C], conj!(tmp))
    conj!(C)
  end
  function mul!(C::AbstractMatrix{T}, Ac::Adjoint{T,SymCUR{T}}, B::AbstractMatrix{T}) where T
    A = parent(Ac)
    tmp = A[:U]'*(A[:C]'*B)
    mul!(C, A[:C], conj!(tmp))
    conj!(C)
  end
  mul!(C::AbstractVector{T}, At::Transpose{T,SymCUR{T}}, B::AbstractVector{T}) where {T} =
    mul!(C, parent(At), B)
  mul!(C::AbstractMatrix{T}, At::Transpose{T,SymCUR{T}}, B::AbstractMatrix{T}) where {T} =
    mul!(C, parent(At), B)

  function mul!(C::AbstractMatrix{T}, Ac::Adjoint{T,SymCUR{T}}, Bc::Adjoint{T,<:AbstractMatrix{T}}) where T
    A = parent(Ac)
    tmp = A[:U]'*(A[:C]'*Bc)
    mul!(C, A[:C], conj!(tmp))
    conj!(C)
  end
  mul!(C::AbstractMatrix{T}, At::Transpose{T,SymCUR{T}}, Bt::Transpose{T,<:AbstractMatrix{T}}) where {T} =
    mul!(C, parent(At), Bt)

  ## SymCUR right-multiplication

  mul!(C::AbstractMatrix{T}, A::AbstractMatrix{T}, B::SymCUR{T}) where {T} =
    mul!(C, (A*B[:C])*B[:U], transpose(B[:C]))

  function mul!!(C::AbstractMatrix{T}, A::AbstractMatrix{T}, Bc::Adjoint{T,SymCUR{T}}) where T
    B = parent(Bc)
    tmp = conj!(A)*B[:C]
    mul!(C, conj!(tmp)*transpose(B[:U]), transpose(B[:C]))
  end  # overwrites A
  function mul!(C::AbstractMatrix{T}, A::AbstractMatrix{T}, Bc::Adjoint{T,SymCUR{T}}) where T
    B = parent(Bc)
    size(A, 1) <= B[:k] && return mul!!(C, copy(A), Bc)
    mul!(C, (A*conj(B[:C]))*B[:U]', B[:C]')
  end
  mul!(C::AbstractMatrix{T}, A::AbstractMatrix{T}, Bt::Transpose{T,SymCUR{T}}) where {T} =
    mul!(C, A, parent(Bt))

  for Adj in (:Adjoint, :Transpose)
    @eval begin
      function mul!(C::AbstractMatrix{T}, Ac::$Adj{T,<:AbstractMatrix{T}}, B::SymCUR{T}) where T
        tmp = Ac * B[:C]
        mul!(C, tmp*B[:U], transpose(B[:C]))
      end
    end
  end

  function mul!(C::AbstractMatrix{T}, Ac::Adjoint{T,<:AbstractMatrix{T}}, Bc::Adjoint{T,SymCUR{T}}) where {T}
    B = parent(Bc)
    A = parent(Ac)
    mul!(C, conj!(transpose(A)*B[:C])*B[:U]', B[:C]')
  end
  mul!(C::AbstractMatrix{T}, At::Transpose{T,<:AbstractMatrix{T}}, Bt::Transpose{T,SymCUR{T}}) where {T} =
    mul!(C, At, parent(Bt))

  # standard operations

  ## left-multiplication
  function *(A::AbstractCUR{TA}, B::AbstractVector{TB}) where {TA,TB}
    T = promote_type(TA, TB)
    AT = convert(AbstractCUR{T}, A)
    BT = (T == TB ? B : convert(Array{T}, B))
    CT = Array{T}(undef, size(A,1))
    mul!(CT, AT, BT)
  end

  for Adj in (:Adjoint, :Transpose)
    @eval function *(Ac::$Adj{TA,AbstractCUR{TA}}, B::AbstractVector{TB}) where {TA,TB}
      A = parent(Ac)
      T = promote_type(TA, TB)
      AT = convert(AbstractCUR{T}, A)
      BT = (T == TB ? B : convert(Array{T}, B))
      CT = Array{T}(undef, size(A,2))
      mul!(CT, $Adj(AT), BT)
    end
  end

  function *(A::AbstractCUR{TA}, B::AbstractMatrix{TB}) where {TA,TB}
    T = promote_type(TA, TB)
    AT = convert(AbstractCUR{T}, A)
    BT = (T == TB ? B : convert(Array{T}, B))
    CT = Array{T}(undef, size(A,1), size(B,2))
    mul!(CT, AT, BT)
  end
  function *(A::AbstractMatrix{TA}, B::AbstractCUR{TB}) where {TA,TB}
    T = promote_type(TA, TB)
    AT = (T == TA ? A : convert(Array{T}, A))
    BT = convert(AbstractCUR{T}, B)
    CT = Array{T}(undef, size(A,1), size(B,2))
    mul!(CT, AT, BT)
  end

  for Adj in (:Adjoint, :Transpose)
    @eval begin
      function *(A::AbstractCUR{TA}, Bc::$Adj{TB,<:AbstractMatrix{TB}}) where {TA,TB}
        B = parent(Bc)
        T = promote_type(TA, TB)
        AT = convert(AbstractCUR{T}, A)
        BT = (T == TB ? B : convert(Array{T}, B))
        CT = Array{T}(undef, size(A,1), size(Bc,2))
        mul!(CT, AT, $Adj(BT))
      end
      function *(Ac::$Adj{TA,<:AbstractCUR{TA}}, B::AbstractMatrix{TB}) where {TA,TB}
        A = parent(Ac)
        T = promote_type(TA, TB)
        AT = convert(AbstractCUR{T}, A)
        BT = (T == TB ? B : convert(Array{T}, B))
        CT = Array{T}(undef, size(Ac,1), size(B,2))
        mul!(CT, $Adj(AT), BT)
      end
      function *(Ac::$Adj{TA,<:AbstractCUR{TA}}, Bc::$Adj{TB,<:AbstractMatrix{TB}}) where {TA,TB}
        A = parent(Ac)
        B = parent(Bc)
        T = promote_type(TA, TB)
        AT = convert(AbstractCUR{T}, A)
        BT = (T == TB ? B : convert(Array{T}, B))
        CT = Array{T}(undef, size(Ac,1), size(Bc,2))
        mul!(CT, $Adj(AT), $Adj(BT))
      end
      function *(A::AbstractMatrix{TA}, Bc::$Adj{TB,<:AbstractCUR{TB}}) where {TA,TB}
        B = parent(Bc)
        T = promote_type(TA, TB)
        AT = (T == TA ? A : convert(Array{T}, A))
        BT = convert(AbstractCUR{T}, B)
        CT = Array{T}(undef, size(A,1), size(Bc,2))
        mul!(CT, AT, $Adj(BT))
      end
      function *(Ac::$Adj{TA,<:AbstractMatrix{TA}}, B::AbstractCUR{TB}) where {TA,TB}
        A = parent(Ac)
        T = promote_type(TA, TB)
        AT = (T == TA ? A : convert(Array{T}, A))
        BT = convert(AbstractCUR{T}, B)
        CT = Array{T}(undef, size(Ac,1), size(B,2))
        mul!(CT, $Adj(AT), BT)
      end
      function *(Ac::$Adj{TA,<:AbstractMatrix{TA}}, Bc::$Adj{TB,<:AbstractCUR{TB}}) where {TA,TB}
        A = parent(Ac)
        B = parent(Bc)
        T = promote_type(TA, TB)
        AT = (T == TA ? A : convert(Array{T}, A))
        BT = convert(AbstractCUR{T}, B)
        CT = Array{T}(undef, size(Ac,1), size(Bc,2))
        mul!(CT, $Adj(AT), $Adj(BT))
      end
    end
  end

# factorization routines

for sfx in ("", "!")
  f = Symbol("curfact", sfx)
  g = Symbol("curfact_none", sfx)
  h = Symbol("cur", sfx)
  @eval begin
    function $f(A::AbstractMatOrLinOp{T}, opts::LRAOptions=LRAOptions(T); args...) where T
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

function curfact_none!(A::AbstractMatrix, opts::LRAOptions)
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
function curfact_none(A::AbstractMatrix, opts::LRAOptions)
  (ishermitian(A) || issymmetric(A)) && return curfact_none!(copy(A), opts)
  m, n = size(A)
  if m >= n
    F = pqrfact_backend!(Matrix(A'), opts)
    rows = F[:p][1:F[:k]]
    F = pqrfact_backend!(A[rows,:], opts)
    cols = F[:p][1:F[:k]]
  else
    F = pqrfact_backend!(copy(A), opts)
    cols = F[:p][1:F[:k]]
    F = pqrfact_backend!(Matrix(A[:,cols]'), opts)
    rows = F[:p][1:F[:k]]
  end
  kr = length(rows)
  kc = length(cols)
  k = min(kr, kc)
  rows = k < kr ? rows[1:k] : rows
  cols = k < kc ? cols[1:k] : cols
  return CURPackedU(rows, cols)
end
