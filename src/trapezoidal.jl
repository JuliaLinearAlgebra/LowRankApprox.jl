#= src/trapezoidal.jl
=#

abstract type Trapezoidal{T} <: AbstractMatrix{T} end

mutable struct LowerTrapezoidal{T} <: Trapezoidal{T}
  data::Matrix{T}
end

mutable struct UpperTrapezoidal{T} <: Trapezoidal{T}
  data::Matrix{T}
end

LowerTrapezoidal(data::AbstractMatrix) = LowerTrapezoidal(Matrix(data))
UpperTrapezoidal(data::AbstractMatrix) = UpperTrapezoidal(Matrix(data))

conj!(A::LowerTrapezoidal) = LowerTrapezoidal(conj!(A.data))
conj!(A::UpperTrapezoidal) = UpperTrapezoidal(conj!(A.data))
conj(A::Trapezoidal) = conj!(copy(A))

convert(::Type{LowerTrapezoidal{T}}, A::LowerTrapezoidal) where {T} =
  LowerTrapezoidal(convert(Array{T}, A.data))
convert(::Type{UpperTrapezoidal{T}}, A::UpperTrapezoidal) where {T} =
  UpperTrapezoidal(convert(Array{T}, A.data))
convert(::Type{Trapezoidal{T}}, A::LowerTrapezoidal) where {T} =
  convert(LowerTrapezoidal{T}, A)
convert(::Type{Trapezoidal{T}}, A::UpperTrapezoidal) where {T} =
  convert(UpperTrapezoidal{T}, A)
convert(::Type{Array}, A::Trapezoidal) = Matrix(A)
convert(::Type{Array{T}}, A::Trapezoidal) where {T} = convert(Array{T}, Matrix(A))
convert(::Type{Matrix}, A::Trapezoidal) = Matrix(A)
convert(::Type{Matrix{T}}, A::Trapezoidal) where {T} = convert(Array{T}, Matrix(A))

copy(A::LowerTrapezoidal) = LowerTrapezoidal(copy(A.data))
copy(A::UpperTrapezoidal) = UpperTrapezoidal(copy(A.data))

adjoint(A::LowerTrapezoidal) = UpperTrapezoidal(A.data')
adjoint(A::UpperTrapezoidal) = LowerTrapezoidal(A.data')
transpose(A::LowerTrapezoidal) = UpperTrapezoidal(transpose(A.data))
transpose(A::UpperTrapezoidal) = LowerTrapezoidal(transpose(A.data))

Matrix(A::LowerTrapezoidal) = tril(A.data)
Matrix(A::UpperTrapezoidal) = triu(A.data)

getindex(A::LowerTrapezoidal{T}, i::Integer, j::Integer) where {T} =
  i >= j ? A.data[i,j] : zero(T)
getindex(A::UpperTrapezoidal{T}, i::Integer, j::Integer) where {T} =
  i <= j ? A.data[i,j] : zero(T)

imag(A::LowerTrapezoidal) = LowerTrapezoidal(imag(A.data))
imag(A::UpperTrapezoidal) = UpperTrapezoidal(imag(A.data))

real(A::LowerTrapezoidal) = LowerTrapezoidal(real(A.data))
real(A::UpperTrapezoidal) = UpperTrapezoidal(real(A.data))

setindex!(A::LowerTrapezoidal, x, i::Integer, j::Integer) =
  i >= j ? (A.data[i,j] = x; A) : throw(BoundsError)
setindex!(A::UpperTrapezoidal, x, i::Integer, j::Integer) =
  i <= j ? (A.data[i,j] = x; A) : throw(BoundsError)

size(A::Trapezoidal, args...) = size(A.data, args...)

# BLAS/LAPACK routines

## LowerTrapezoidal left-multiplication
  function mul!(y::AbstractVector{T}, A::LowerTrapezoidal{T}, x::AbstractVector{T}) where T
    m, n = size(A)
    y[1:n] = x
    lmul!(LowerTriangular(view(A.data,1:n,:)), view(y,1:n))
    mul!(view(y,n+1:m), view(A.data,n+1:m,:), x)
    y
  end
  function mul!(C::AbstractMatrix{T}, A::LowerTrapezoidal{T}, B::AbstractMatrix{T}) where T
    m, n = size(A)
    C[1:n,:] = B
    lmul!(LowerTriangular(view(A.data,1:n,:)), view(C,1:n,:))
    mul!(view(C,n+1:m,:), view(A.data,n+1:m,:), B)
    C
  end
  for Adj in (:Adjoint, :Transpose)
      @eval function mul!(C::AbstractMatrix{T}, A::LowerTrapezoidal{T}, B::$Adj{T,<:AbstractMatrix{T}}) where T
        m, n = size(A)
        C[1:n,:] = B
        lmul!(LowerTriangular(view(A.data,1:n,:)), view(C,1:n,:))
        mul!(view(C,n+1:m,:), view(A.data,n+1:m,:), B)
        C
      end
  end
  for (Adj, g) in ((:Adjoint, :adjoint!), (:Transpose, :transpose!))
    @eval function mul!(C::AbstractMatrix{T}, A::$Adj{T,LowerTrapezoidal{T}}, B::AbstractMatrix{T}) where T
        m, n = size(A)
        $g(view(C,1:n,:), B)
        lmul!(LowerTriangular(view(A.data,1:n,:)), view(C,1:n,:))
        mul!(view(C,n+1:m,:), $Adj(view(A.data,n+1:m,:)), B)
        C
    end
  end

  for (Adj, trans) in ((:Adjoint, 'C'), (:Transpose, 'T'))
    @eval begin
      function mul!(y::AbstractVector{T}, adjA::$Adj{T,LowerTrapezoidal{T}}, x::AbstractVector{T}) where T<:BlasFloat
        A = parent(adjA)
        m, n = size(A)
        copyto!(y, view(x,1:n))
        BLAS.trmv!('L', $trans, 'N', view(A.data,1:n,:), y)
        BLAS.gemv!($trans, one(T), view(A.data,n+1:m,:), view(x,n+1:m), one(T), y)
      end
      function mul!(C::AbstractMatrix{T}, adjA::$Adj{T,LowerTrapezoidal{T}}, B::AbstractMatrix{T}) where T<:BlasFloat
        A = parent(adjA)
        m, n = size(A)
        copyto!(C, view(B,1:n,:))
        BLAS.trmm!('L', 'L', $trans, 'N', one(T), view(A.data,1:n,:), C)
        BLAS.gemm!($trans, 'N', one(T), view(A.data,n+1:m,:), view(B,n+1:m,:), one(T), C)
      end
    end
  end

  for (Adj, g, trans) in ((:Adjoint, :adjoint!, 'C'),
                        (:Transpose, :transpose!,  'T'))
    @eval begin
      function mul!(C::AbstractMatrix{T}, adjA::$Adj{T,LowerTrapezoidal{T}}, adjB::$Adj{T,<:AbstractMatrix{T}}) where T<:BlasFloat
        A = parent(adjA)
        B = parent(adjB)
        m, n = size(A)
        $g(C, view(B,:,1:n))
        BLAS.trmm!(
          'L', 'L', $trans, 'N', one(T), view(A.data,1:n,:), view(C,1:n,:))
        BLAS.gemm!(
          $trans, $trans, one(T), view(A.data,n+1:m,:), view(B,:,n+1:m), one(T), C)
      end
    end
  end

  ## LowerTrapezoidal right-multiplication

  function mul!(C::AbstractMatrix{T}, A::AbstractMatrix{T}, B::LowerTrapezoidal{T}) where T<:BlasFloat
    m, n = size(B)
    copyto!(C, view(A,:,1:n))
    rmul!(C, LowerTriangular(view(B.data,1:n,:)))
    BLAS.gemm!('N', 'N', one(T), view(A,:,n+1:m), view(B.data,n+1:m,:), one(T), C)
  end

  for (Adj, trans) in ((:Adjoint, 'C'), (:Transpose, 'T'))
    @eval begin
      function mul!(C::AbstractMatrix{T}, A::AbstractMatrix{T}, adjB::$Adj{T,LowerTrapezoidal{T}}) where T<:BlasFloat
        B = parent(adjB)
        m, n = size(B)
        copyto!(view(C,:,1:n), A)
        BLAS.trmm!(
          'R', 'L', $trans, 'N', one(T), view(B.data,1:n,:), view(C,:,1:n))
        mul!(view(C,:,n+1:m), A, $Adj(view(B.data,n+1:m,:)))
        C
      end
    end
  end

  for (Adj, g, trans) in ((:Adjoint, :adjoint!, 'C'),
                        (:Transpose, :transpose!,  'T'))
    @eval begin
      function mul!(
          C::AbstractMatrix{T}, adjA::$Adj{T,<:AbstractMatrix{T}}, B::LowerTrapezoidal{T}) where T<:BlasFloat
        A = parent(adjA)
        m, n = size(B)
        $g(C, view(A,1:n,:))
        BLAS.trmm!('R', 'L', 'N', 'N', one(T), view(B.data,1:n,:), C)
        BLAS.gemm!(
          $trans, 'N', one(T), view(A,n+1:m,:), view(B.data,n+1:m,:), one(T), C)
      end
    end
  end

  for (Adj, g, trans) in ((:Adjoint, :adjoint!, 'C'),
                        (:Transpose, :transpose!,  'T'))
    @eval begin
      function mul!(
          C::AbstractMatrix{T}, adjA::$Adj{T,<:AbstractMatrix{T}}, adjB::$Adj{T,LowerTrapezoidal{T}}) where T<:BlasFloat
        A = parent(adjA)
        B = parent(adjB)
        m, n = size(B)
        $g(view(C,:,1:n), A)
        BLAS.trmm!(
          'R', 'L', $trans, 'N', one(T), view(B.data,1:n,:), view(C,:,1:n))
        mul!(view(C,:,n+1:m), $Adj(A), $Adj(view(B.data,n+1:m,:)))
        C
      end
    end
  end

  ## UpperTrapezoidal left-multiplication

  function mul!(
      y::AbstractVector{T}, A::UpperTrapezoidal{T}, x::AbstractVector{T}) where T<:BlasFloat
    m, n = size(A)
    copyto!(y, view(x,1:m))
    lmul!(UpperTriangular(view(A.data,:,1:m)), y)
    BLAS.gemv!('N', one(T), view(A.data,:,m+1:n), view(x,m+1:n), one(T), y)
  end
  function mul!(
      C::AbstractMatrix{T}, A::UpperTrapezoidal{T}, B::AbstractMatrix{T}) where T<:BlasFloat
    m, n = size(A)
    copyto!(C, view(B,1:m,:))
    lmul!(UpperTriangular(view(A.data,:,1:m)), C)
    BLAS.gemm!(
      'N', 'N', one(T), view(A.data,:,m+1:n), view(B,m+1:n,:), one(T), C)
  end

  for (Adj, g, trans) in ((:Adjoint, :adjoint!, 'C'),
                        (:Transpose, :transpose!,  'T'))
    @eval begin
      function mul!(C::AbstractMatrix{T}, A::UpperTrapezoidal{T}, adjB::$Adj{T,<:AbstractMatrix{T}}) where T<:BlasFloat
        B = parent(adjB)
        m, n = size(A)
        $g(C, view(B,:,1:m))
        lmul!(UpperTriangular(view(A.data,:,1:m)), C)
        BLAS.gemm!(
          'N', $trans, one(T), view(A.data,:,m+1:n), view(B,:,m+1:n), one(T), C)
      end
    end
  end

  for (Adj, trans) in ((:Adjoint, 'C'), (:Transpose, 'T'))
    @eval begin
      function mul!(y::AbstractVector{T}, adjA::$Adj{T,UpperTrapezoidal{T}}, x::AbstractVector{T}) where T<:BlasFloat
        A = parent(adjA)
        m, n = size(A)
        y[1:m] = x
        BLAS.trmv!('U', $trans, 'N', view(A.data,:,1:m), view(y,1:m))
        mul!(view(y,m+1:n), $Adj(view(A.data,:,m+1:n)), x)
        y
      end
      function mul!(C::AbstractMatrix{T}, adjA::$Adj{T,UpperTrapezoidal{T}}, B::AbstractMatrix{T}) where T<:BlasFloat
        A = parent(adjA)
        m, n = size(A)
        C[1:m,:] = B
        BLAS.trmm!(
          'L', 'U', $trans, 'N', one(T), view(A.data,:,1:m), view(C,1:m,:))
        mul!(view(C,m+1:n,:), $Adj(view(A.data,:,m+1:n)), B)
        C
      end
    end
  end

  for (Adj, g, trans) in ((:Adjoint, :adjoint!, 'C'),
                        (:Transpose, :transpose!,  'T'))
    @eval begin
      function mul!(C::AbstractMatrix{T}, adjA::$Adj{T,UpperTrapezoidal{T}}, adjB::$Adj{T,AbstractMatrix{T}}) where T<:BlasFloat
        A = parent(adjA)
        B = parent(adjB)
        m, n = size(A)
        $g(view(C,1:m,:), B)
        BLAS.trmm!(
          'L', 'U', $trans, 'N', one(T), view(A.data,:,1:m), view(C,1:m,:))
        mul!(view(C,m+1:n,:), $Adj(view(A.data,:,m+1:n)), $Adj(B))
        C
      end
    end
  end

  ## UpperTrapezoidal right-multiplication

  function mul!(
      C::AbstractMatrix{T}, A::AbstractMatrix{T}, B::UpperTrapezoidal{T}) where T
    m, n = size(B)
    C[:,1:m] = A
    rmul!(view(C,:,1:m), UpperTriangular(view(B.data,:,1:m)))
    mul!(view(C,:,m+1:n), A, view(B.data,:,m+1:n))
    C
  end

  for (Adj, trans) in ((:Adjoint, 'C'), (:Transpose, 'T'))
    @eval begin
      function mul!(
          C::AbstractMatrix{T}, A::AbstractMatrix{T}, adjB::$Adj{T,UpperTrapezoidal{T}}) where T<:BlasFloat
        B = parent(adjB)
        m, n = size(B)
        copyto!(C, view(A,:,1:m))
        BLAS.trmm!('R', 'U', $trans, 'N', one(T), view(B.data,:,1:m), C)
        BLAS.gemm!(
          'N', $trans, one(T), view(A,:,m+1:n), view(B.data,:,m+1:n), one(T), C)
      end
    end
  end

  for (Adj, g) in ((:Adjoint, :adjoint!), (:Transpose, :transpose!))
    @eval begin
      function mul!(
          C::AbstractMatrix{T}, adjA::$Adj{T,<:AbstractMatrix{T}}, B::UpperTrapezoidal{T}) where T
        A = parent(adjA)
        m, n = size(B)
        $g(view(C,:,1:m), A)
        rmul!(view(C,:,1:m), UpperTriangular(view(B.data,:,1:m)))
        mul!(view(C,:,m+1:n), $Adj(A), view(B.data,:,m+1:n))
        C
      end
    end
  end

  for (Adj, g, trans) in ((:Adjoint, :adjoint!, 'C'),
                        (:Transpose, :transpose!,  'T'))
    @eval begin
      function mul!(C::AbstractMatrix{T}, adjA::$Adj{T,<:AbstractMatrix{T}}, adjB::$Adj{T,UpperTrapezoidal{T}}) where T<:BlasFloat
        A = parent(adjA)
        B = parent(adjB)
        m, n = size(B)
        mul!(C, view(A,1:m,:))
        BLAS.trmm!('R', 'U', $trans, 'N', one(T), view(B.data,:,1:m), C)
        BLAS.gemm!(
          $trans, $trans, one(T), view(A,m+1:n,:), view(B.data,:,m+1:n),
          one(T), C)
      end
    end
  end

  # standard operations

  ## left-multiplication

  function *(A::Trapezoidal{TA}, B::AbstractVector{TB}) where {TA,TB}
    T = promote_type(TA, TB)
    AT = convert(Trapezoidal{T}, A)
    BT = (T == TB ? B : convert(Array{T}, B))
    CT = Array{T}(undef, size(A,1))
    mul!(CT, AT, BT)
  end

  function *(A::Trapezoidal{TA}, B::AbstractMatrix{TB}) where {TA,TB}
    T = promote_type(TA, TB)
    AT = convert(Trapezoidal{T}, A)
    BT = (T == TB ? B : convert(Array{T}, B))
    CT = Array{T}(undef, size(A,1), size(B,2))
    mul!(CT, AT, BT)
  end
  function *(A::AbstractMatrix{TA}, B::Trapezoidal{TB}) where {TA,TB}
    T = promote_type(TA, TB)
    AT = (T == TA ? A : convert(Array{T}, A))
    BT = convert(Trapezoidal{T}, B)
    CT = Array{T}(undef, size(A,1), size(B,2))
    mul!(CT, AT, BT)
  end

  for Adj in (:Adjoint, :Transpose)
    @eval begin
      function *(adjA::$Adj{TA,<:Trapezoidal{TA}}, B::AbstractVector{TB}) where {TA,TB}
        A = parent(adjA)
        T = promote_type(TA, TB)
        AT = convert(Trapezoidal{T}, A)
        BT = (T == TB ? B : convert(Array{T}, B))
        CT = Array{T}(undef, size(A,2))
        mul!(CT, $Adj(AT), BT)
      end
      function *(adjA::$Adj{TA,<:Trapezoidal{TA}}, B::AbstractMatrix{TB}) where {TA,TB}
        A = parent(adjA)
        T = promote_type(TA, TB)
        AT = convert(Trapezoidal{T}, A)
        BT = (T == TB ? B : convert(Array{T}, B))
        CT = Array{T}(undef, size(A,2), size(B,2))
        mul!(CT, $Adj(AT), BT)
      end
      function *(A::Trapezoidal{TA}, adjB::$Adj{TA,<:AbstractMatrix{TB}}) where {TA,TB}
        B = parent(adjB)
        T = promote_type(TA, TB)
        AT = convert(Trapezoidal{T}, A)
        BT = (T == TB ? B : convert(Array{T}, B))
        CT = Array{T}(undef, size(A,1), size(B,1))
        mul!(CT, AT, $Adj(BT))
      end
      function *(adjA::$Adj{TA,<:Trapezoidal{TA}}, adjB::$Adj{TA,<:AbstractMatrix{TB}}) where {TA,TB}
        A = parent(adjA)
        B = parent(adjB)
        T = promote_type(TA, TB)
        AT = convert(Trapezoidal{T}, A)
        BT = (T == TB ? B : convert(Array{T}, B))
        CT = Array{T}(undef, size(A,1), size(B,1))
        mul!(CT, $Adj(AT), $Adj(BT))
      end

      function *(adjA::$Adj{TA,<:AbstractMatrix{TA}}, B::Trapezoidal{TB}) where {TA,TB}
        A = parent(adjA)
        T = promote_type(TA, TB)
        AT = (T == TA ? A : convert(Array{T}, A))
        BT = convert(Trapezoidal{T}, B)
        CT = Array{T}(undef, size(A,2), size(B,2))
        mul!(CT, $Adj(AT), BT)
      end
      function *(A::AbstractMatrix{TA}, adjB::$Adj{TA,<:Trapezoidal{TB}}) where {TA,TB}
        B = parent(adjB)
        T = promote_type(TA, TB)
        AT = (T == TA ? A : convert(Array{T}, A))
        BT = convert(Trapezoidal{T}, B)
        CT = Array{T}(undef, size(A,1), size(B,1))
        mul!(CT, AT, $Adj(BT))
      end
      function *(adjA::$Adj{TA,<:AbstractMatrix{TA}}, adjB::$Adj{TA,<:Trapezoidal{TB}}) where {TA,TB}
        A = parent(adjA)
        B = parent(adjB)
        T = promote_type(TA, TB)
        AT = (T == TA ? A : convert(Array{T}, A))
        BT = convert(Trapezoidal{T}, B)
        CT = Array{T}(undef, size(A,2), size(B,2))
        mul!(CT, $Adj(AT), $Adj(BT))
      end
    end
  end
