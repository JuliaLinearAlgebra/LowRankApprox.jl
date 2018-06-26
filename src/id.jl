#= src/id.jl

References:

  H. Cheng, Z. Gimbutas, P.G. Martinsson, V. Rokhlin. On the compression of low
    rank matrices. SIAM J. Sci. Comput. 26 (4): 1389-1404, 2005.

  E. Liberty, F. Woolfe, P.-G. Martinsson V. Rokhlin, M. Tygert. Randomized
    algorithms for the low-rank approximation of matrices. Proc. Natl. Acad.
    Sci. U.S.A. 104 (51): 20167-20172, 2007.
=#

# IDPackedV

mutable struct IDPackedV{S} <: Factorization{S}
  sk::Vector{Int}
  rd::Vector{Int}
  T::Matrix{S}
end

conj!(A::IDPackedV) = IDPackedV(A.sk, A.rd, conj!(A.T))
conj(A::IDPackedV) = IDPackedV(A.sk, A.rd, conj(A.T))

convert(::Type{IDPackedV{T}}, A::IDPackedV) where {T} =
  IDPackedV(A.sk, A.rd, convert(Array{T}, A.T))
convert(::Type{Factorization{T}}, A::IDPackedV) where {T} = convert(IDPackedV{T}, A)
convert(::Type{Array}, A::IDPackedV) = full(A)
convert(::Type{Array{T}}, A::IDPackedV) where {T} = convert(Array{T}, full(A))

copy(A::IDPackedV) = IDPackedV(copy(A.sk), copy(A.rd), copy(A.T))
if VERSION < v"0.7-"
  function full!(trans::Symbol, A::StridedMatrix{T}, V::IDPackedV{T}) where T
    chktrans(trans)
    k, n = size(V)
    if trans == :n
      size(A) == (k, n) || throw(DimensionMismatch)
      @inbounds for j = 1:k
        @simd for i = 1:k
          A[i,j] = i == j ? 1 : 0
        end
      end
      A[:,k+1:n] = V[:T]
      A_mul_Bc!(A, V[:P])
    else
      size(A) == (n, k) || throw(DimensionMismatch)
      @inbounds for j = 1:k
        @simd for i = 1:k
          A[i,j] = i == j ? 1 : 0
        end
      end
      adjoint!(view(A,k+1:n,:), V[:T])
      mul!(V[:P], A)
    end
    A
  end
else
  function full!(trans::Symbol, A::StridedMatrix{T}, V::IDPackedV{T}) where T
    chktrans(trans)
    k, n = size(V)
    if trans == :n
      size(A) == (k, n) || throw(DimensionMismatch)
      @inbounds for j = 1:k
        @simd for i = 1:k
          A[i,j] = i == j ? 1 : 0
        end
      end
      A[:,k+1:n] = V[:T]
      rmul!(A, V[:P]')
    else
      size(A) == (n, k) || throw(DimensionMismatch)
      @inbounds for j = 1:k
        @simd for i = 1:k
          A[i,j] = i == j ? 1 : 0
        end
      end
      adjoint!(view(A,k+1:n,:), V[:T])
      lmul!(V[:P], A)
    end
    A
  end
end
full!(A::StridedMatrix{T}, V::IDPackedV{T}) where {T} = full!(:n, A, V)
function full(trans::Symbol, A::IDPackedV{T}) where T
  chktrans(trans)
  k, n = size(A)
  if trans == :n  B = Array{T}(undef, k, n)
  else            B = Array{T}(undef, n, k)
  end
  full!(trans, B, A)
end
full(A::IDPackedV) = full(:n, A)

if VERSION ≥ v"0.7-"
  adjoint(A::IDPackedV) = Adjoint(A)
  transpose(A::IDPackedV) = Transpose(A)
end


function getindex(A::IDPackedV, d::Symbol)
  if     d == :P   return ColumnPermutation(A[:p])
  elseif d == :T   return A.T
  elseif d == :k   return length(A.sk)
  elseif d == :p   return [A.sk; A.rd]
  elseif d == :rd  return A.rd
  elseif d == :sk  return A.sk
  else             throw(KeyError(d))
  end
end

ishermitian(::IDPackedV) = false
issym(::IDPackedV) = false

isreal(::IDPackedV{T}) where {T} = T <: Real

ndims(::IDPackedV) = 2

size(A::IDPackedV) = (size(A.T,1), sum(size(A.T)))
size(A::IDPackedV, dim::Integer) =
  (dim == 1 ? size(A.T,1) : (dim == 2 ? sum(size(A.T)) : 1))

## BLAS/LAPACK multiplication routines
if VERSION < v"0.7-"
  ### left-multiplication
  function mul!!(
      y::StridedVector{T}, A::IDPackedV{T}, x::StridedVector{T}) where T<:BlasFloat
    k, n = size(A)
    Ac_mul_B!(A[:P], x)
    copyto!(y, view(x,1:k))
    BLAS.gemv!('N', one(T), A[:T], view(x,k+1:n), one(T), y)
  end  # overwrites x
  function mul!!(
      C::StridedMatrix{T}, A::IDPackedV{T}, B::StridedMatrix{T}) where T<:BlasFloat
    k, n = size(A)
    Ac_mul_B!(A[:P], B)
    copyto!(C, view(B,1:k,:))
    BLAS.gemm!('N', 'N', one(T), A[:T], view(B,k+1:n,:), one(T), C)
  end  # overwrites B
  mul!(C::StridedVecOrMat{T}, A::IDPackedV{T}, B::StridedVecOrMat{T}) where {T} =
    mul!!(C, A, copy(B))

  for (f!, g) in ((:A_mul_Bc!, :Ac_mul_Bc), (:A_mul_Bt!, :At_mul_Bt))
    @eval begin
      function $f!(
          C::StridedMatrix{T}, A::IDPackedV{T}, B::StridedMatrix{T}) where T<:BlasFloat
        k, n = size(A)
        tmp = $g(A[:P], B)
        copyto!(C, view(tmp,1:k,:))
        BLAS.gemm!('N', 'N', one(T), A[:T], view(tmp,k+1:n,:), one(T), C)
      end
    end
  end

  for f in (:Ac_mul_B!, :At_mul_B!)
    @eval begin
      function $f(
          C::StridedVecOrMat{T}, A::IDPackedV{T}, B::StridedVecOrMat{T}) where T
        k, n = size(A)
        copyto!(view(C,1:k,:), B)
        $f(view(C,k+1:n,:), A[:T], B)
        mul!(A[:P], C)
      end
    end
  end

  for (f, g) in ((:Ac_mul_Bc!, :adjoint!), (:At_mul_Bt!, :transpose!))
    @eval begin
      function $f(C::StridedMatrix{T}, A::IDPackedV{T}, B::StridedMatrix{T}) where T
        k, n = size(A)
        $g(view(C,1:k,:), B)
        $f(view(C,k+1:n,:), A[:T], B)
        mul!(A[:P], C)
      end
    end
  end

  ### right-multiplication

  function mul!(C::StridedMatrix{T}, A::StridedMatrix{T}, B::IDPackedV{T}) where T
    k, n = size(B)
    copyto!(view(C,:,1:k), A)
    mul!(view(C,:,k+1:n), A, B[:T])
    A_mul_Bc!(C, B[:P])
  end

  for (f!, trans) in ((:A_mul_Bc!, 'C'), (:A_mul_Bt!, 'T'))
    f!! = Symbol(f!, "!")
    @eval begin
      function $f!!(
          C::StridedMatrix{T}, A::StridedMatrix{T}, B::IDPackedV{T}) where T<:BlasFloat
        k, n = size(B)
        mul!(A, B[:P])
        copyto!(C, view(A,:,1:k))
        BLAS.gemm!('N', $trans, one(T), view(A,:,k+1:n), B[:T], one(T), C)
      end  # overwrites A
      $f!(C::StridedMatrix{T}, A::StridedMatrix{T}, B::IDPackedV{T}) where {T} =
        $f!!(C, copy(A), B)
    end
  end

  for (f, g, h) in ((:Ac_mul_B!, :adjoint!, :A_mul_Bc!),
                    (:At_mul_B!, :transpose!,  :A_mul_Bt!))
    @eval begin
      function $f(C::StridedMatrix{T}, A::StridedMatrix{T}, B::IDPackedV{T}) where T
        k, n = size(B)
        $g(view(C,:,1:k), A)
        $f(view(C,:,k+1:n), A, B[:T])
        $h(C, B[:P])
      end
    end
  end

  for (f!, g, trans) in ((:Ac_mul_Bc!, :Ac_mul_B, 'C'),
                         (:At_mul_Bt!, :At_mul_B, 'T'))
    @eval begin
      function $f!(C::StridedMatrix{T}, A::StridedMatrix{T}, B::IDPackedV{T}) where T<:BlasFloat
        k, n = size(B)
        tmp = $g(A, B[:P])
        copyto!(C, view(tmp,:,1:k))
        BLAS.gemm!('N', $trans, one(T), view(tmp,:,k+1:n), B[:T], one(T), C)
      end
    end
  end
else
  ### left-multiplication
  function mul!!(y::StridedVector{T}, A::IDPackedV{T}, x::StridedVector{T}) where T<:BlasFloat
    k, n = size(A)
    lmul!(A[:P]', x)
    copyto!(y, view(x,1:k))
    BLAS.gemv!('N', one(T), A[:T], view(x,k+1:n), one(T), y)
  end  # overwrites x
  function mul!!(C::StridedMatrix{T}, A::IDPackedV{T}, B::StridedMatrix{T}) where T<:BlasFloat
    k, n = size(A)
    lmul!(A[:P]', B)
    copyto!(C, view(B,1:k,:))
    BLAS.gemm!('N', 'N', one(T), A[:T], view(B,k+1:n,:), one(T), C)
  end  # overwrites B
  mul!(C::StridedVecOrMat{T}, A::IDPackedV{T}, B::StridedVecOrMat{T}) where {T} =
    mul!!(C, A, copy(B))

  for Adj in (:Transpose, :Adjoint)
    @eval begin
      function mul!(C::StridedMatrix{T}, A::IDPackedV{T}, Bc::$Adj{T,<:StridedMatrix{T}}) where T<:BlasFloat
        k, n = size(A)
        tmp = $Adj(A[:P]) * Bc
        copyto!(C, view(tmp,1:k,:))
        BLAS.gemm!('N', 'N', one(T), A[:T], view(tmp,k+1:n,:), one(T), C)
      end
      function mul!(C::StridedVecOrMat{T}, Ac::$Adj{T,IDPackedV{T}}, B::StridedVecOrMat{T}) where T
        A = parent(Ac)
        k, n = size(A)
        copyto!(view(C,1:k,:), B)
        mul!(view(C,k+1:n,:), $Adj(A[:T]), B)
        lmul!(A[:P], C)
      end
    end
  end

  for (Adj, adj!) in ((:Transpose, :transpose!), (:Adjoint,:adjoint!))
    @eval begin
      function mul!(C::StridedMatrix{T}, Ac::$Adj{T,IDPackedV{T}}, Bc::$Adj{T,<:StridedMatrix{T}}) where T
        A = parent(Ac)
        B = parent(Bc)
        k, n = size(A)
        $adj!(view(C,1:k,:), B)
        mul!(view(C,k+1:n,:), $Adj(A[:T]), $Adj(B))
        lmul!(A[:P], C)
      end
    end
  end

  ### right-multiplication

  function mul!(C::StridedMatrix{T}, A::StridedMatrix{T}, B::IDPackedV{T}) where T
    k, n = size(B)
    copyto!(view(C,:,1:k), A)
    mul!(view(C,:,k+1:n), A, B[:T])
    rmul!(C, B[:P]')
  end

  for (Adj, trans) in ((:Adjoint, 'C'), (:Transpose, 'T'))
    @eval begin
      function mul!!(C::StridedMatrix{T}, A::StridedMatrix{T}, Bc::$Adj{T,IDPackedV{T}}) where T<:BlasFloat
        B = parent(Bc)
        k, n = size(B)
        rmul!(A, B[:P])
        copyto!(C, view(A,:,1:k))
        BLAS.gemm!('N', $trans, one(T), view(A,:,k+1:n), B[:T], one(T), C)
      end  # overwrites A
      mul!(C::StridedMatrix{T}, A::StridedMatrix{T}, Bc::$Adj{T,IDPackedV{T}}) where {T} =
        mul!!(C, copy(A), Bc)
    end
  end

  for (Adj, adj!) in ((:Transpose, :transpose!), (:Adjoint,:adjoint!))
    @eval begin
      function mul!(C::StridedMatrix{T}, Ac::$Adj{T,<:StridedMatrix{T}}, B::IDPackedV{T}) where T
        A = parent(Ac)
        k, n = size(B)
        $adj!(view(C,:,1:k), A)
        mul!(view(C,:,k+1:n), $Adj(A), B[:T])
        rmul!(C, $Adj(B[:P]))
      end
    end
  end

  for (Adj, trans) in ((:Adjoint, 'C'), (:Transpose, 'T'))
    @eval begin
      function mul!(C::StridedMatrix{T}, Ac::$Adj{T,<:StridedMatrix{T}}, Bc::$Adj{T,IDPackedV{T}}) where T<:BlasFloat
        B = parent(Bc)
        k, n = size(B)
        tmp = Ac * B[:P]
        copyto!(C, view(tmp,:,1:k))
        BLAS.gemm!('N', $trans, one(T), view(tmp,:,k+1:n), B[:T], one(T), C)
      end
    end
  end
end

# ID

mutable struct ID{S} <: Factorization{S}
  sk::Vector{Int}
  rd::Vector{Int}
  C::Matrix{S}
  T::Matrix{S}
  ID{S}(sk::Vector{Int}, rd::Vector{Int}, C::Matrix{S}, T::Matrix{S}) where S =
    new{S}(sk, rd, C, T)
end
ID{S}(sk::Vector{Int}, rd::Vector{Int}, C::AbstractMatrix{S}, T::AbstractMatrix{S}) where S =
  ID{S}(sk, rd, convert(Matrix{S}, C), convert(Matrix{S}, T))

function ID(trans::Symbol, A::AbstractMatOrLinOp{T}, V::IDPackedV{T}) where T
  chktrans(trans)
  ID{T}(V.sk, V.rd, getcols(trans, A, V.sk), V.T)
end
ID(trans::Symbol, A::AbstractMatOrLinOp, sk, rd, T) =
  ID(trans, A, IDPackedV(sk, rd, T))
ID(A::AbstractMatOrLinOp, args...) = ID(:n, A, args...)
ID(sk::Vector{Int}, rd::Vector{Int}, C::Matrix{S}, T::Matrix{S}) where S =
  ID{S}(sk, rd, C, T)
ID(A, args...) = ID(LinOp(A), args...)

conj!(A::ID) = ID(A.sk, A.rd, conj!(A.C), conj!(A.T))
conj(A::ID) = ID(A.sk, A.rd, conj(A.C), conj(A.T))

convert(::Type{ID{T}}, A::ID) where {T} =
  ID(A.sk, A.rd, convert(Array{T}, A.C), convert(Array{T}, A.T))
convert(::Factorization{T}, A::ID) where {T} = convert(ID{T}, A)
convert(::Type{Array}, A::ID) = full(A)
convert(::Type{Array{T}}, A::ID) where {T} = convert(Array{T}, full(A))

copy(A::ID) = ID(copy(A.sk), copy(A.rd), copy(A.C), copy(A.T))

full(A::ID) = A[:C]*A[:V]

if VERSION ≥ v"0.7-"
  adjoint(A::ID) = Adjoint(A)
  transpose(A::ID) = Transpose(A)
end


function getindex(A::ID{T}, d::Symbol) where T
  if     d == :C   return A.C
  elseif d == :P   return ColumnPermutation(A[:p])
  elseif d == :T   return A.T
  elseif d == :V   return IDPackedV(A.sk, A.rd, A.T)
  elseif d == :k   return length(A.sk)
  elseif d == :p   return [A.sk; A.rd]
  elseif d == :rd  return A.rd
  elseif d == :sk  return A.sk
  else             throw(KeyError(d))
  end
end

ishermitian(::ID) = false
issym(::ID) = false

isreal(::ID{T}) where {T} = T <: Real

ndims(::ID) = 2

size(A::ID) = (size(A.C,1), sum(size(A.T)))
size(A::ID, dim::Integer) =
  (dim == 1 ? size(A.C,1) : (dim == 2 ? sum(size(A.T)) : 1))

## BLAS/LAPACK multiplication routines

### left-multiplication

function mul!!(y::StridedVector{T}, A::ID{T}, x::StridedVector{T}) where T
  tmp = Array{T}(undef, A[:k])
  mul!!(tmp, A[:V], x)
  mul!(y, A[:C], tmp)
end  # overwrites x
function mul!!(C::StridedMatrix{T}, A::ID{T}, B::StridedMatrix{T}) where T
  tmp = Array{T}(undef, A[:k], size(B,2))
  mul!!(tmp, A[:V], B)
  mul!(C, A[:C], tmp)
end  # overwrites B
mul!(C::StridedVecOrMat{T}, A::ID{T}, B::StridedVecOrMat{T}) where {T} =
  mul!!(C, A, copy(B))

if VERSION < v"0.7-"
  for f in (:A_mul_Bc, :A_mul_Bt)
    f! = Symbol(f, "!")
    @eval begin
      function $f!(C::StridedMatrix{T}, A::ID{T}, B::StridedMatrix{T}) where T
        tmp = $f(A[:V], B)
        mul!(C, A[:C], tmp)
      end
    end
  end

  for f in (:Ac_mul_B, :At_mul_B)
    f! = Symbol(f, "!")
    @eval begin
      function $f!(C::StridedVecOrMat{T}, A::ID{T}, B::StridedVecOrMat{T}) where T
        tmp = $f(A[:C], B)
        $f!(C, A[:V], tmp)
      end
    end
  end

  for (f, g!) in ((:Ac_mul_Bc, :Ac_mul_B!), (:At_mul_Bt, :At_mul_B!))
    f! = Symbol(f, "!")
    @eval begin
      function $f!(C::StridedMatrix{T}, A::ID{T}, B::StridedMatrix{T}) where T
        tmp = $f(A[:C], B)
        $g!(C, A[:V], tmp)
      end
    end
  end

  ### right-multiplication

  mul!(C::StridedMatrix{T}, A::StridedMatrix{T}, B::ID{T}) where {T} =
    mul!(C, A*B[:C], B[:V])

  for f! in (:A_mul_Bc!, :A_mul_Bt!)
    f!! = Symbol(f!, "!")
    @eval begin
      function $f!!(C::StridedMatrix{T}, A::StridedMatrix{T}, B::ID{T}) where T
        tmp = Array{T}(undef, size(A,1), B[:k])
        $f!!(tmp, A, B[:V])
        $f!(C, tmp, B[:C])
      end  # overwrites A
      $f!(C::StridedMatrix{T}, A::StridedMatrix{T}, B::ID{T}) where {T} =
        $f!!(C, copy(A), B)
    end
  end

  for f in (:Ac_mul_B, :At_mul_B)
    f! = Symbol(f, "!")
    @eval begin
      function $f!(C::StridedMatrix{T}, A::StridedMatrix{T}, B::ID{T}) where T
        tmp = $f(A, B[:C])
        mul!(C, tmp, B[:V])
      end
    end
  end

  for (f, g!) in ((:Ac_mul_Bc, :A_mul_Bc!), (:At_mul_Bt, :A_mul_Bt!))
    f! = Symbol(f, "!")
    @eval begin
      function $f!(C::StridedMatrix{T}, A::StridedMatrix{T}, B::ID{T}) where T
        tmp = $f(A, B[:V])
        $g!(C, tmp, B[:C])
      end
    end
  end

  # standard operations

  ## left-multiplication

  for (f, f!, i) in ((:*,        :mul!,  1),
                     (:Ac_mul_B, :Ac_mul_B!, 2),
                     (:At_mul_B, :At_mul_B!, 2))
    for t in (:IDPackedV, :ID)
      @eval begin
        function $f(A::$t{TA}, B::StridedVector{TB}) where {TA,TB}
          T = promote_type(TA, TB)
          AT = convert($t{T}, A)
          BT = (T == TB ? B : convert(Array{T}, B))
          CT = Array{T}(undef, size(A,$i))
          $f!(CT, AT, BT)
        end
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
    for t in (:IDPackedV, :ID)
      @eval begin
        function $f(A::$t{TA}, B::StridedMatrix{TB}) where {TA,TB}
          T = promote_type(TA, TB)
          AT = convert($t{T}, A)
          BT = (T == TB ? B : convert(Array{T}, B))
          CT = Array{T}(undef, size(A,$i), size(B,$j))
          $f!(CT, AT, BT)
        end
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
    for t in (:IDPackedV, :ID)
      @eval begin
        function $f(A::StridedMatrix{TA}, B::$t{TB}) where {TA,TB}
          T = promote_type(TA, TB)
          AT = (T == TA ? A : convert(Array{T}, A))
          BT = convert($t{T}, B)
          CT = Array{T}(undef, size(A,$i), size(B,$j))
          $f!(CT, AT, BT)
        end
      end
    end
  end
else
  for Adj in (:Transpose, :Adjoint)
    @eval begin
      function mul!(C::StridedMatrix{T}, A::ID{T}, Bc::$Adj{T,<:StridedMatrix{T}}) where T
        tmp = A[:V] * Bc
        mul!(C, A[:C], tmp)
      end
      function mul!(C::StridedVecOrMat{T}, Ac::$Adj{T,ID{T}}, B::StridedVecOrMat{T}) where T
        A = parent(Ac)
        tmp = $Adj(A[:C]) * B
        mul!(C, $Adj(A[:V]), tmp)
      end
      function mul!(C::StridedMatrix{T}, Ac::$Adj{T,ID{T}}, Bc::$Adj{T,<:StridedMatrix{T}}) where T
        A = parent(Ac)
        tmp = $Adj(A[:C]) * Bc
        mul!(C, $Adj(A[:V]), tmp)
      end
        ### right-multiplication
      function mul!!(C::StridedMatrix{T}, A::StridedMatrix{T}, Bc::$Adj{T,ID{T}}) where T
        B = parent(Bc)
        tmp = Array{T}(undef, size(A,1), B[:k])
        mul!!(tmp, A, $Adj(B[:V]))
        mul!(C, tmp, $Adj(B[:C]))
      end  # overwrites A
      mul!(C::StridedMatrix{T}, A::StridedMatrix{T}, Bc::$Adj{T,ID{T}}) where {T} =
        mul!!(C, copy(A), Bc)
      function mul!(C::StridedMatrix{T}, Ac::$Adj{T,<:StridedMatrix{T}}, B::ID{T}) where T
        tmp = Ac * B[:C]
        mul!(C, tmp, B[:V])
      end
      function mul!(C::StridedMatrix{T}, Ac::$Adj{T,<:StridedMatrix{T}}, Bc::$Adj{T,ID{T}}) where T
        B = parent(Bc)
        tmp = Ac * $Adj(B[:V])
        mul!(C, tmp, $Adj(B[:C]))
      end
    end
  end

  ### right-multiplication

  mul!(C::StridedMatrix{T}, A::StridedMatrix{T}, B::ID{T}) where {T} =
    mul!(C, A*B[:C], B[:V])



  # standard operations

  ## left-multiplication
  for t in (:IDPackedV, :ID)
    @eval begin
      function *(A::$t{TA}, B::StridedVector{TB}) where {TA,TB}
        T = promote_type(TA, TB)
        AT = convert($t{T}, A)
        BT = (T == TB ? B : convert(Array{T}, B))
        CT = Array{T}(undef, size(A,1))
        mul!(CT, AT, BT)
      end
      function *(A::$t{TA}, B::StridedMatrix{TB}) where {TA,TB}
        T = promote_type(TA, TB)
        AT = convert($t{T}, A)
        BT = (T == TB ? B : convert(Array{T}, B))
        CT = Array{T}(undef, size(A,1), size(B,2))
        mul!(CT, AT, BT)
      end
      function *(A::StridedMatrix{TA}, B::$t{TB}) where {TA,TB}
        T = promote_type(TA, TB)
        AT = (T == TA ? A : convert(Array{T}, A))
        BT = convert($t{T}, B)
        CT = Array{T}(undef, size(A,1), size(B,2))
        mul!(CT, AT, BT)
      end
    end
    for Adj in (:Transpose, :Adjoint)
      @eval begin
        function *(Ac::$Adj{TA,$t{TA}}, B::StridedVector{TB}) where {TA,TB}
          A = parent(Ac)
          T = promote_type(TA, TB)
          AT = convert($t{T}, A)
          BT = (T == TB ? B : convert(Array{T}, B))
          CT = Array{T}(undef, size(Ac,1))
          mul!(CT, $Adj(AT), BT)
        end
        function *(A::$t{TA}, Bc::$Adj{TB,<:StridedMatrix{TB}}) where {TA,TB}
          B = parent(Bc)
          T = promote_type(TA, TB)
          AT = convert($t{T}, A)
          BT = (T == TB ? B : convert(Array{T}, B))
          CT = Array{T}(undef, size(A,1), size(Bc,2))
          mul!(CT, AT, $Adj(BT))
        end
        function *(Ac::$Adj{TA,$t{TA}}, B::StridedMatrix{TB}) where {TA,TB}
          A = parent(Ac)
          T = promote_type(TA, TB)
          AT = convert($t{T}, A)
          BT = (T == TB ? B : convert(Array{T}, B))
          CT = Array{T}(undef, size(Ac,1), size(B,2))
          mul!(CT, $Adj(AT), BT)
        end
        function *(Ac::$Adj{TA,$t{TA}}, Bc::$Adj{TB,<:StridedMatrix{TB}}) where {TA,TB}
          A = parent(Ac)
          B = parent(Bc)
          T = promote_type(TA, TB)
          AT = convert($t{T}, A)
          BT = (T == TB ? B : convert(Array{T}, B))
          CT = Array{T}(undef, size(Ac,1), size(Bc,2))
          mul!(CT, $Adj(AT), $Adj(BT))
        end
        function *(A::StridedMatrix{TA}, Bc::$Adj{TB,$t{TB}}) where {TA,TB}
          B = parent(Bc)
          T = promote_type(TA, TB)
          AT = (T == TA ? A : convert(Array{T}, A))
          BT = convert($t{T}, B)
          CT = Array{T}(undef, size(A,1), size(Bc,2))
          mul!(CT, AT, $Adj(BT))
        end
        function *(Ac::$Adj{TA,<:StridedMatrix{TA}}, B::$t{TB}) where {TA,TB}
          A = parent(Ac)
          T = promote_type(TA, TB)
          AT = (T == TA ? A : convert(Array{T}, A))
          BT = convert($t{T}, B)
          CT = Array{T}(undef, size(Ac,1), size(B,2))
          mul!(CT, $Adj(AT), BT)
        end
        function *(Ac::$Adj{TA,<:StridedMatrix{TA}}, Bc::$Adj{TB,$t{TB}}) where {TA,TB}
          A = parent(Ac)
          B = parent(Bc)
          T = promote_type(TA, TB)
          AT = (T == TA ? A : convert(Array{T}, A))
          BT = convert($t{T}, B)
          CT = Array{T}(undef, size(Ac,1), size(Bc,2))
          mul!(CT, $Adj(AT), $Adj(BT))
        end
      end
    end
  end
end

# factorization routines

for sfx in ("", "!")
  f = Symbol("idfact", sfx)
  g = Symbol("pqrfact", sfx)
  h = Symbol("id", sfx)
  @eval begin
    function $f(
        trans::Symbol, A::AbstractMatOrLinOp{T}, opts::LRAOptions=LRAOptions(T);
        args...) where T
      opts = copy(opts; args...)
      opts.pqrfact_retval = "t"
      chkopts!(opts, A)
      if opts.sketch == :none
        F = $g(trans, A, opts)
      else
        F = sketchfact(:left, trans, A, opts)
      end
      k = F[:k]
      IDPackedV(F.p[1:k], F.p[k+1:end], get(F.T))
    end
    $f(trans::Symbol, A, args...; kwargs...) =
      $f(trans, LinOp(A), args...; kwargs...)
    $f(A, args...; kwargs...) = $f(:n, A, args...; kwargs...)

    function $h(trans::Symbol, A, args...; kwargs...)
      V = $f(trans, A, args...; kwargs...)
      V.sk, V.rd, V.T
    end
    $h(A, args...; kwargs...) = $h(:n, A, args...; kwargs...)
  end
end
