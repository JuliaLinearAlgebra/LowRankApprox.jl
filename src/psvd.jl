#= src/psvd.jl
=#

mutable struct PartialSVD{T,Tr<:Real} <: Factorization{T}
  U::Matrix{T}
  S::Vector{Tr}
  Vt::Matrix{T}

  function PartialSVD{T,Tr}(U::Matrix{T}, S::Vector{Tr}, Vt::Matrix{T}) where {T,Tr<:Real}
    size(U,2) == size(Vt,1) == length(S) || throw(DimensionMismatch("$(size(U)), $(length(S)), $(size(Vt)) not compatible"))
    new{T,Tr}(U, S, Vt)
  end
end

PartialSVD(U::AbstractMatrix{T}, S::AbstractVector{Tr}, Vt::AbstractMatrix{T}) where {T,Tr<:Real} =
  PartialSVD{T,Tr}(Matrix(U), Vector(S), Matrix(Vt))

conj!(A::PartialSVD) = PartialSVD(conj!(A.U), A.S, conj!(A.Vt))
conj(A::PartialSVD) = PartialSVD(conj(A.U), A.S, conj(A.Vt))

function convert(::Type{PartialSVD{T}}, A::PartialSVD) where T
  Tr = real(T)
  PartialSVD(
    convert(Array{T}, A.U), convert(Array{Tr}, A.S), convert(Array{T}, A.Vt))
end
convert(::Type{Factorization{T}}, A::PartialSVD) where {T} = convert(PartialSVD{T}, A)
convert(::Type{Factorization{T}}, A::PartialSVD{T,Tr}) where {T,Tr<:Real} = convert(PartialSVD{T}, A)
convert(::Type{Array}, A::PartialSVD) = Matrix(A)
convert(::Type{Array{T}}, A::PartialSVD) where {T} = convert(Array{T}, Matrix(A))

copy(A::PartialSVD) = PartialSVD(copy(A.U), copy(A.S), copy(A.Vt))

adjoint!(A::PartialSVD) = PartialSVD(A.Vt', A.S, A.U')
adjoint(A::PartialSVD) = PartialSVD(A.Vt', copy(A.S), A.U')
transpose!(A::PartialSVD) = PartialSVD(transpose(A.Vt), A.S, transpose(A.U))
transpose(A::PartialSVD) = PartialSVD(transpose(A.Vt), copy(A.S), transpose(A.U))

Matrix(A::PartialSVD) = (A[:U]*Diagonal(A[:S]))*A[:Vt]

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

  mul!(y::AbstractVector{T}, A::PartialSVD{T}, x::AbstractVector{T}) where {T} =
    mul!(y, A[:U], scalevec!(A[:S], A[:Vt]*x))
  mul!(C::AbstractMatrix{T}, A::PartialSVD{T}, B::AbstractMatrix{T}) where {T} =
    mul!(C, A[:U], lmul!(Diagonal(A[:S]), A[:Vt]*B))

  ## right-multiplication

  mul!(C::AbstractMatrix{T}, A::AbstractMatrix{T}, B::PartialSVD{T}) where {T} =
    mul!(C, rmul!(A*B[:U], Diagonal(B[:S])), B[:Vt])

  for Adj in (:Adjoint, :Transpose)
    @eval begin
      function mul!(C::AbstractMatrix{T}, A::PartialSVD{T}, Bc::$Adj{T,<:AbstractMatrix{T}}) where T
        tmp = A[:Vt] * Bc
        lmul!(Diagonal(A[:S]), tmp)
        mul!(C, A[:U], tmp)
      end
      function mul!(y::AbstractVector{T}, Ac::$Adj{T,<:PartialSVD{T}}, x::AbstractVector{T}) where T
        A = parent(Ac)
        tmp = $Adj(A[:U]) * x
        scalevec!(A[:S], tmp)
        mul!(y, $Adj(A[:Vt]), tmp)
      end
      function mul!(C::AbstractMatrix{T}, Ac::$Adj{T,<:PartialSVD{T}}, B::AbstractMatrix{T}) where T
        A = parent(Ac)
        tmp = $Adj(A[:U]) * B
        lmul!(Diagonal(A[:S]), tmp)
        mul!(C, $Adj(A[:Vt]), tmp)
      end
      function mul!(C::AbstractMatrix{T}, Ac::$Adj{T,<:PartialSVD{T}}, Bc::$Adj{T,<:AbstractMatrix{T}}) where T
        A = parent(Ac)
        tmp = $Adj(A[:U]) * Bc
        lmul!(Diagonal(A[:S]), tmp)
        mul!(C, $Adj(A[:Vt]), tmp)
      end
      ## right-multiplication
      function mul!(C::AbstractMatrix{T}, A::AbstractMatrix{T}, Bc::$Adj{T,<:PartialSVD{T}}) where T
        B = parent(Bc)
        tmp = A * $Adj(B[:Vt])
        rmul!(tmp, Diagonal(B[:S]))
        mul!(C, tmp, $Adj(B[:U]))
      end
      function mul!(C::AbstractMatrix{T}, Ac::$Adj{T,<:AbstractMatrix{T}}, B::PartialSVD{T}) where T
        tmp = Ac * B[:U]
        rmul!(tmp, Diagonal(B[:S]))
        mul!(C, tmp, B[:Vt])
      end
      function mul!(C::AbstractMatrix{T}, Ac::$Adj{T,<:AbstractMatrix{T}}, Bc::$Adj{T,<:PartialSVD{T}}) where T
        tmp = Ac * $Adj(B[:Vt])
        rmul!(tmp, Diagonal(B[:S]))
        mul!(C, tmp, $Adj(B[:U]))
      end
    end
  end

  ## left-division (pseudoinverse left-multiplication)
  ldiv!(y::AbstractVector{T}, A::PartialSVD{T}, x::AbstractVector{T}) where {T} =
    mul!(y, A[:Vt]', iscalevec!(A[:S], A[:U]'*x))
  ldiv!(C::AbstractMatrix{T}, A::PartialSVD{T}, B::AbstractMatrix{T}) where {T} =
    mul!(C, A[:Vt]', iscale!(A[:S], A[:U]'*B))

  # standard operations

  ## left-multiplication
  function *(A::PartialSVD{TA}, B::AbstractVector{TB}) where {TA,TB}
    T = promote_type(TA, TB)
    AT = convert(PartialSVD{T}, A)
    BT = (T == TB ? B : convert(Array{T}, B))
    CT = Array{T}(undef, size(A,1))
    mul!(CT, AT, BT)
  end

  for Adj in (:Transpose, :Adjoint)
    @eval begin
      function *(Ac::$Adj{TA,<:PartialSVD{TA}}, B::AbstractVector{TB}) where {TA,TB}
        A = parent(Ac)
        T = promote_type(TA, TB)
        AT = convert(PartialSVD{T}, A)
        BT = (T == TB ? B : convert(Array{T}, B))
        CT = Array{T}(undef, size(Ac,1))
        mul!(CT, $Adj(AT), BT)
      end
    end
  end

  function *(A::PartialSVD{TA}, B::AbstractMatrix{TB}) where {TA,TB}
    T = promote_type(TA, TB)
    AT = convert(PartialSVD{T}, A)
    BT = (T == TB ? B : convert(Array{T}, B))
    CT = Array{T}(undef, size(A,1), size(B,2))
    mul!(CT, AT, BT)
  end
  function *(A::AbstractMatrix{TA}, B::PartialSVD{TB}) where {TA,TB}
    T = promote_type(TA, TB)
    AT = (T == TA ? A : convert(Array{T}, A))
    BT = convert(PartialSVD{T}, B)
    CT = Array{T}(undef, size(A,1), size(B,2))
    mul!(CT, AT, BT)
  end
  for Adj in (:Transpose, :Adjoint)
    @eval begin
      function *(A::PartialSVD{TA}, Bc::$Adj{TB,<:AbstractMatrix{TB}}) where {TA,TB}
        B = parent(Bc)
        T = promote_type(TA, TB)
        AT = convert(PartialSVD{T}, A)
        BT = (T == TB ? B : convert(Array{T}, B))
        CT = Array{T}(undef, size(A,1), size(Bc,2))
        mul!(CT, AT, $Adj(BT))
      end
      function *(Ac::$Adj{TA,<:PartialSVD{TA}}, B::AbstractMatrix{TB}) where {TA,TB}
        A = parent(Ac)
        T = promote_type(TA, TB)
        AT = convert(PartialSVD{T}, A)
        BT = (T == TB ? B : convert(Array{T}, B))
        CT = Array{T}(undef, size(Ac,1), size(B,2))
        mul!(CT, $Adj(AT), BT)
      end
      function *(Ac::$Adj{TA,<:PartialSVD{TA}}, Bc::$Adj{TB,<:AbstractMatrix{TB}}) where {TA,TB}
        A = parent(Ac)
        B = parent(Bc)
        T = promote_type(TA, TB)
        AT = convert(PartialSVD{T}, A)
        BT = (T == TB ? B : convert(Array{T}, B))
        CT = Array{T}(undef, size(Ac,1), size(Bc,2))
        mul!(CT, $Adj(AT), $Adj(BT))
      end
      function *(A::AbstractMatrix{TA}, Bc::$Adj{TB,<:PartialSVD{TB}}) where {TA,TB}
        B = parent(Bc)
        T = promote_type(TA, TB)
        AT = (T == TA ? A : convert(Array{T}, A))
        BT = convert(PartialSVD{T}, B)
        CT = Array{T}(undef, size(A,1), size(Bc,2))
        mul!(CT, AT, $Adj(BT))
      end
      function *(Ac::$Adj{TA,<:AbstractMatrix{TA}}, B::PartialSVD{TB}) where {TA,TB}
        A = parent(Ac)
        T = promote_type(TA, TB)
        AT = (T == TA ? A : convert(Array{T}, A))
        BT = convert(PartialSVD{T}, B)
        CT = Array{T}(undef, size(Ac,1), size(B,2))
        mul!(CT, $Adj(AT), BT)
      end
      function *(Ac::$Adj{TA,<:AbstractMatrix{TA}}, Bc::$Adj{TB,<:PartialSVD{TB}}) where {TA,TB}
        A = parent(Ac)
        B = parent(Bc)
        T = promote_type(TA, TB)
        AT = (T == TA ? A : convert(Array{T}, A))
        BT = convert(PartialSVD{T}, B)
        CT = Array{T}(undef, size(Ac,1), size(Bc,2))
        mul!(CT, $Adj(AT), $Adj(BT))
      end
    end
  end


## left-division
function \(A::PartialSVD{TA}, B::AbstractVector{TB}) where {TA,TB}
  T = promote_type(TA, TB)
  AT = convert(PartialSVD{T}, A)
  BT = (T == TB ? B : convert(Array{T}, B))
  CT = Array{T}(undef, size(A,2))
  ldiv!(CT, AT, BT)
end
function \(A::PartialSVD{TA}, B::AbstractMatrix{TB}) where {TA,TB}
  T = promote_type(TA, TB)
  AT = convert(PartialSVD{T}, A)
  BT = (T == TB ? B : convert(Array{T}, B))
  CT = Array{T}(undef, size(A,2), size(B,2))
  ldiv!(CT, AT, BT)
end

# factorization routines

function psvdfact(
    A::AbstractMatOrLinOp{T}, opts::LRAOptions=LRAOptions(T); args...) where T
  opts = isempty(args) ? opts : copy(opts; args...)
  m, n = size(A)
  if m >= n
    V = idfact(:n, A, opts)
    Q,R = qr!(getcols(:n, A, V[:sk]))
    Ũ, σ, Ṽ = svd!(R*V)
    k = psvdrank(σ, opts)
    if k < V[:k]
      U  = Q*view(Ũ,:,1:k)
      S  = σ[1:k]
      Vt = Ṽ'[1:k,:]
    else
      U  = Q*Ũ
      S  = σ
      Vt = Ṽ'
    end
  else
    V = idfact(:c, A, opts)
    Q,R = qr!(getcols(:c, A, V[:sk]))
    Ũ, σ, Ṽ = svd!(V'*R')
    k = psvdrank(σ, opts)
    if k < V[:k]
      U  = Ũ[:,1:k]
      S  = σ[1:k]
      Vt = view(Ṽ',1:k,:)*Q'
    else
      U  = Ũ
      S  = σ
      Vt = Ṽ'*Q'
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
    Q,R = qr!(getcols(:n, A, V[:sk]))
    s = svdvals!(R*V)
  else
    V = idfact(:c, A, opts)
    Q,R = qr!(getcols(:c, A, V[:sk]))
    s = svdvals!(V'*R')
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
