#= src/trapezoidal.jl
=#

abstract Trapezoidal{T} <: AbstractMatrix{T}

type LowerTrapezoidal{T} <: Trapezoidal{T}
  data::Matrix{T}
end

type UpperTrapezoidal{T} <: Trapezoidal{T}
  data::Matrix{T}
end

conj!(A::LowerTrapezoidal) = LowerTrapezoidal(conj!(A.data))
conj!(A::UpperTrapezoidal) = UpperTrapezoidal(conj!(A.data))
conj(A::Trapezoidal) = conj!(copy(A))

convert{T}(::Type{LowerTrapezoidal{T}}, A::LowerTrapezoidal) =
  LowerTrapezoidal(convert(Array{T}, A.data))
convert{T}(::Type{UpperTrapezoidal{T}}, A::UpperTrapezoidal) =
  UpperTrapezoidal(convert(Array{T}, A.data))
convert{T}(::Type{Trapezoidal{T}}, A::LowerTrapezoidal) =
  convert(LowerTrapezoidal{T}, A)
convert{T}(::Type{Trapezoidal{T}}, A::UpperTrapezoidal) =
  convert(UpperTrapezoidal{T}, A)
convert(::Type{Array}, A::Trapezoidal) = full(A)
convert{T}(::Type{Array{T}}, A::Trapezoidal) = convert(Array{T}, full(A))

copy(A::LowerTrapezoidal) = LowerTrapezoidal(copy(A.data))
copy(A::UpperTrapezoidal) = UpperTrapezoidal(copy(A.data))

ctranspose(A::LowerTrapezoidal) = UpperTrapezoidal(A.data')
ctranspose(A::UpperTrapezoidal) = LowerTrapezoidal(A.data')
transpose(A::LowerTrapezoidal) = UpperTrapezoidal(A.data.')
transpose(A::UpperTrapezoidal) = LowerTrapezoidal(A.data.')

full(A::LowerTrapezoidal) = tril(A.data)
full(A::UpperTrapezoidal) = triu(A.data)

getindex{T}(A::LowerTrapezoidal{T}, i::Integer, j::Integer) =
  i >= j ? A.data[i,j] : zero(T)
getindex{T}(A::UpperTrapezoidal{T}, i::Integer, j::Integer) =
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

function A_mul_B!{T}(
    y::StridedVector{T}, A::LowerTrapezoidal{T}, x::StridedVector{T})
  m, n = size(A)
  y[1:n] = x
  A_mul_B!(LowerTriangular(sub(A.data,1:n,:)), sub(y,1:n))
  A_mul_B!(sub(y,n+1:m), sub(A.data,n+1:m,:), x)
  y
end
function A_mul_B!{T}(
    C::StridedMatrix{T}, A::LowerTrapezoidal{T}, B::StridedMatrix{T})
  m, n = size(A)
  C[1:n,:] = B
  A_mul_B!(LowerTriangular(sub(A.data,1:n,:)), sub(C,1:n,:))
  A_mul_B!(sub(C,n+1:m,:), sub(A.data,n+1:m,:), B)
  C
end

for (f, g) in ((:A_mul_Bc!, :ctranspose!), (:A_mul_Bt!, :transpose!))
  @eval begin
    function $f{T}(
        C::StridedMatrix{T}, A::LowerTrapezoidal{T}, B::StridedMatrix{T})
      m, n = size(A)
      $g(sub(C,1:n,:), B)
      A_mul_B!(LowerTriangular(sub(A.data,1:n,:)), sub(C,1:n,:))
      $f(sub(C,n+1:m,:), sub(A.data,n+1:m,:), B)
      C
    end
  end
end

for (f, trans) in ((:Ac_mul_B!, 'C'), (:At_mul_B!, 'T'))
  @eval begin
    function $f{T<:BlasFloat}(
        y::StridedVector{T}, A::LowerTrapezoidal{T}, x::StridedVector{T})
      m, n = size(A)
      copy!(y, sub(x,1:n))
      BLAS.trmv!('L', $trans, 'N', sub(A.data,1:n,:), y)
      BLAS.gemv!($trans, one(T), sub(A.data,n+1:m,:), sub(x,n+1:m), one(T), y)
    end
    function $f{T<:BlasFloat}(
        C::StridedMatrix{T}, A::LowerTrapezoidal{T}, B::StridedMatrix{T})
      m, n = size(A)
      copy!(C, sub(B,1:n,:))
      BLAS.trmm!('L', 'L', $trans, 'N', one(T), sub(A.data,1:n,:), C)
      BLAS.gemm!(
        $trans, 'N', one(T), sub(A.data,n+1:m,:), sub(B,n+1:m,:), one(T), C)
    end
  end
end

for (f, g, trans) in ((:Ac_mul_Bc!, :ctranspose!, 'C'),
                      (:At_mul_Bt!, :transpose!,  'T'))
  @eval begin
    function $f{T<:BlasFloat}(
        C::StridedMatrix{T}, A::LowerTrapezoidal{T}, B::StridedMatrix{T})
      m, n = size(A)
      $g(C, sub(B,:,1:n))
      BLAS.trmm!('L', 'L', $trans, 'N', one(T), sub(A.data,1:n,:), sub(C,1:n,:))
      BLAS.gemm!(
        $trans, $trans, one(T), sub(A.data,n+1:m,:), sub(B,:,n+1:m), one(T), C)
    end
  end
end

## LowerTrapezoidal right-multiplication

function A_mul_B!{T<:BlasFloat}(
    C::StridedMatrix{T}, A::StridedMatrix{T}, B::LowerTrapezoidal{T})
  m, n = size(B)
  copy!(C, sub(A,:,1:n))
  A_mul_B!(C, LowerTriangular(sub(B.data,1:n,:)))
  BLAS.gemm!('N', 'N', one(T), sub(A,:,n+1:m), sub(B.data,n+1:m,:), one(T), C)
end

for (f, trans) in ((:A_mul_Bc!, 'C'), (:A_mul_Bt!, 'T'))
  @eval begin
    function $f{T<:BlasFloat}(
        C::StridedMatrix{T}, A::StridedMatrix{T}, B::LowerTrapezoidal{T})
      m, n = size(B)
      copy!(sub(C,:,1:n), A)
      BLAS.trmm!('R', 'L', $trans, 'N', one(T), sub(B.data,1:n,:), sub(C,:,1:n))
      $f(sub(C,:,n+1:m), A, sub(B.data,n+1:m,:))
      C
    end
  end
end

for (f, g, trans) in ((:Ac_mul_B!, :ctranspose!, 'C'),
                      (:At_mul_B!, :transpose!,  'T'))
  @eval begin
    function $f{T<:BlasFloat}(
        C::StridedMatrix{T}, A::StridedMatrix{T}, B::LowerTrapezoidal{T})
      m, n = size(B)
      $g(C, sub(A,1:n,:))
      BLAS.trmm!('R', 'L', 'N', 'N', one(T), sub(B.data,1:n,:), C)
      BLAS.gemm!(
        $trans, 'N', one(T), sub(A,n+1:m,:), sub(B.data,n+1:m,:), one(T), C)
    end
  end
end

for (f, g, trans) in ((:Ac_mul_Bc!, :ctranspose!, 'C'),
                      (:At_mul_Bt!, :transpose!,  'T'))
  @eval begin
    function $f{T<:BlasFloat}(
        C::StridedMatrix{T}, A::StridedMatrix{T}, B::LowerTrapezoidal{T})
      m, n = size(B)
      $g(sub(C,:,1:n), A)
      BLAS.trmm!('R', 'L', $trans, 'N', one(T), sub(B.data,1:n,:), sub(C,:,1:n))
      $f(sub(C,:,n+1:m), A, sub(B.data,n+1:m,:))
      C
    end
  end
end

## UpperTrapezoidal left-multiplication

function A_mul_B!{T<:BlasFloat}(
    y::StridedVector{T}, A::UpperTrapezoidal{T}, x::StridedVector{T})
  m, n = size(A)
  copy!(y, sub(x,1:m))
  A_mul_B!(UpperTriangular(sub(A.data,:,1:m)), y)
  BLAS.gemv!('N', one(T), sub(A.data,:,m+1:n), sub(x,m+1:n), one(T), y)
end
function A_mul_B!{T<:BlasFloat}(
    C::StridedMatrix{T}, A::UpperTrapezoidal{T}, B::StridedMatrix{T})
  m, n = size(A)
  copy!(C, sub(B,1:m,:))
  A_mul_B!(UpperTriangular(sub(A.data,:,1:m)), C)
  BLAS.gemm!('N', 'N', one(T), sub(A.data,:,m+1:n), sub(B,m+1:n,:), one(T), C)
end

for (f, g, trans) in ((:A_mul_Bc!, :ctranspose!, 'C'),
                      (:A_mul_Bt!, :transpose!,  'T'))
  @eval begin
    function $f{T<:BlasFloat}(
        C::StridedMatrix{T}, A::UpperTrapezoidal{T}, B::StridedMatrix{T})
      m, n = size(A)
      $g(C, sub(B,:,1:m))
      A_mul_B!(UpperTriangular(sub(A.data,:,1:m)), C)
      BLAS.gemm!(
        'N', $trans, one(T), sub(A.data,:,m+1:n), sub(B,:,m+1:n), one(T), C)
    end
  end
end

for (f, trans) in ((:Ac_mul_B!, 'C'), (:At_mul_B!, 'T'))
  @eval begin
    function $f{T<:BlasFloat}(
        y::StridedVector{T}, A::UpperTrapezoidal{T}, x::StridedVector{T})
      m, n = size(A)
      y[1:m] = x
      BLAS.trmv!('U', $trans, 'N', sub(A.data,:,1:m), sub(y,1:m))
      $f(sub(y,m+1:n), sub(A.data,:,m+1:n), x)
      y
    end
    function $f{T<:BlasFloat}(
        C::StridedMatrix{T}, A::UpperTrapezoidal{T}, B::StridedMatrix{T})
      m, n = size(A)
      C[1:m,:] = B
      BLAS.trmm!('L', 'U', $trans, 'N', one(T), sub(A.data,:,1:m), sub(C,1:m,:))
      $f(sub(C,m+1:n,:), sub(A.data,:,m+1:n), B)
      C
    end
  end
end

for (f, g, trans) in ((:Ac_mul_Bc!, :ctranspose!, 'C'),
                      (:At_mul_Bt!, :transpose!,  'T'))
  @eval begin
    function $f{T<:BlasFloat}(
        C::StridedMatrix{T}, A::UpperTrapezoidal{T}, B::StridedMatrix{T})
      m, n = size(A)
      $g(sub(C,1:m,:), B)
      BLAS.trmm!('L', 'U', $trans, 'N', one(T), sub(A.data,:,1:m), sub(C,1:m,:))
      $f(sub(C,m+1:n,:), sub(A.data,:,m+1:n), B)
      C
    end
  end
end

## UpperTrapezoidal right-multiplication

function A_mul_B!{T}(
    C::StridedMatrix{T}, A::StridedMatrix{T}, B::UpperTrapezoidal{T})
  m, n = size(B)
  C[:,1:m] = A
  A_mul_B!(sub(C,:,1:m), UpperTriangular(sub(B.data,:,1:m)))
  A_mul_B!(sub(C,:,m+1:n), A, sub(B.data,:,m+1:n))
  C
end

for (f, trans) in ((:A_mul_Bc!, 'C'), (:A_mul_Bt!, 'T'))
  @eval begin
    function $f{T<:BlasFloat}(
        C::StridedMatrix{T}, A::StridedMatrix{T}, B::UpperTrapezoidal{T})
      m, n = size(B)
      copy!(C, sub(A,:,1:m))
      BLAS.trmm!('R', 'U', $trans, 'N', one(T), sub(B.data,:,1:m), C)
      BLAS.gemm!(
        'N', $trans, one(T), sub(A,:,m+1:n), sub(B.data,:,m+1:n), one(T), C)
    end
  end
end

for (f, g) in ((:Ac_mul_B!, :ctranspose!), (:At_mul_B!, :transpose!))
  @eval begin
    function $f{T}(
        C::StridedMatrix{T}, A::StridedMatrix{T}, B::UpperTrapezoidal{T})
      m, n = size(B)
      $g(sub(C,:,1:m), A)
      A_mul_B!(sub(C,:,1:m), UpperTriangular(sub(B.data,:,1:m)))
      $f(sub(C,:,m+1:n), A, sub(B.data,:,m+1:n))
      C
    end
  end
end

for (f, g, trans) in ((:Ac_mul_Bc!, :ctranspose!, 'C'),
                      (:At_mul_Bt!, :transpose!,  'T'))
  @eval begin
    function $f{T<:BlasFloat}(
        C::StridedMatrix{T}, A::StridedMatrix{T}, B::UpperTrapezoidal{T})
      m, n = size(B)
      $g(C, sub(A,1:m,:))
      BLAS.trmm!('R', 'U', $trans, 'N', one(T), sub(B.data,:,1:m), C)
      BLAS.gemm!(
        $trans, $trans, one(T), sub(A,m+1:n,:), sub(B.data,:,m+1:n), one(T), C)
    end
  end
end

# standard operations

## left-multiplication

for (f, f!, i) in ((:*,        :A_mul_B!,  1),
                   (:Ac_mul_B, :Ac_mul_B!, 2),
                   (:At_mul_B, :At_mul_B!, 2))
  @eval begin
    function $f{TA,TB}(A::Trapezoidal{TA}, B::StridedVector{TB})
      T = promote_type(TA, TB)
      AT = convert(Trapezoidal{T}, A)
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
    function $f{TA,TB}(A::Trapezoidal{TA}, B::StridedMatrix{TB})
      T = promote_type(TA, TB)
      AT = convert(Trapezoidal{T}, A)
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
    function $f{TA,TB}(A::StridedMatrix{TA}, B::Trapezoidal{TB})
      T = promote_type(TA, TB)
      AT = (T == TA ? A : convert(Array{T}, A))
      BT = convert(Trapezoidal{T}, B)
      CT = Array(T, size(A,$i), size(B,$j))
      $f!(CT, AT, BT)
    end
  end
end