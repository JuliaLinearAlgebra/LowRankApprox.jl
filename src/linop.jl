#= src/linop.jl
=#

abstract type AbstractLinearOperator{T} end
const AbstractLinOp = AbstractLinearOperator
const AbstractMatOrLinOp{T} = Union{AbstractMatrix{T}, AbstractLinOp{T}}

mutable struct LinearOperator{T} <: AbstractLinOp{T}
  m::Int
  n::Int
  mul!::Function
  mulc!::Function
  _tmp::Nullable{Array{T}}
end
const LinOp = LinearOperator

mutable struct HermitianLinearOperator{T} <: AbstractLinOp{T}
  n::Int
  mul!::Function
  _tmp::Nullable{Array{T}}
end
const HermLinOp = HermitianLinearOperator

function LinOp(A)
  try
    ishermitian(A) && return HermLinOp(A)
  end
  T = eltype(A)
  m, n = size(A)
  mul!  = (y, _, x) ->  A_mul_B!(y, A, x)
  mulc! = (y, _, x) -> Ac_mul_B!(y, A, x)
  LinOp{T}(m, n, mul!, mulc!, nothing)
end

function HermLinOp(A)
  T = eltype(A)
  m, n = size(A)
  m == n || throw(DimensionMismatch)
  mul! = (y, _, x) ->  A_mul_B!(y, A, x)
  HermLinOp{T}(n, mul!, nothing)
end

convert(::Type{Array}, A::AbstractLinOp) = full(A)
convert(::Type{Array{T}}, A::AbstractLinOp) where {T} = convert(Array{T}, full(A))

adjoint(A::LinOp{T}) where {T} = LinOp{T}(A.n, A.m, A.mulc!, A.mul!, nothing)
adjoint(A::HermLinOp) = A

eltype(::AbstractLinOp{T}) where {T} = T

full(A::AbstractLinOp{T}) where {T} = A*eye(T, size(A,2))

getindex(A::AbstractLinOp, ::Colon, ::Colon) = full(A)
function getindex(A::AbstractLinOp{T}, ::Colon, cols) where T
  k = length(cols)
  S = zeros(T, size(A,2), k)
  for i = 1:k
    S[cols[i],i] = 1
  end
  A*S
end
function getindex(A::AbstractLinOp{T}, rows, ::Colon) where T
  k = length(rows)
  S = zeros(T, size(A,1), k)
  for i = 1:k
    S[rows[i],i] = 1
  end
  (A'*S)'
end
function getindex(A::AbstractLinOp{T}, rows, cols) where T
  if length(rows) >= length(cols)  return A[:,cols][rows,:]
  else                             return A[rows,:][:,cols]
  end
end

ishermitian(::LinOp) = false
ishermitian(::HermLinOp) = true
issymmetric(::LinOp) = false
issymmetric(A::HermLinOp) = isreal(A)

isreal(::AbstractLinOp{T}) where {T} = T <: Real

size(A::LinOp) = (A.m, A.n)
size(A::LinOp, dim::Integer) = dim == 1 ? A.m : (dim == 2 ? A.n : 1)
size(A::HermLinOp) = (A.n, A.n)
size(A::HermLinOp, dim::Integer) = (dim == 1 || dim == 2) ? A.n : 1

function transpose(A::LinOp{T}) where T
  n, m = size(A)
  mul!  = (y, L, x) -> (A.mulc!(y, L, conj(x)); conj!(y))
  mulc! = (y, L, x) -> ( A.mul!(y, L, conj(x)); conj!(y))
  LinOp{T}(m, n, mul!, mulc!, nothing)
end
function transpose(A::HermLinOp{T}) where T
  n = size(A, 1)
  mul! = (y, L, x) -> (A.mul!(y, L, conj(x)); conj!(y))
  HermLinOp{T}(n, mul!, nothing)
end

# matrix multiplication

A_mul_B!(C, A::AbstractLinOp, B::AbstractVecOrMat) = A.mul!(C, A, B)
Ac_mul_B!(C, A::LinOp, B::AbstractVecOrMat) = A.mulc!(C, A, B)
Ac_mul_B!(C, A::HermLinOp, B::AbstractVecOrMat) = A_mul_B!(C, A, B)

A_mul_B!(C, A::AbstractMatrix, B::AbstractLinOp) = adjoint!(C, B'*A')
A_mul_Bc!(C, A::AbstractMatrix, B::AbstractLinOp) = adjoint!(C, B*A')

*(A::AbstractLinOp{T}, x::AbstractVector) where {T} =
  (y = Array{T}(size(A,1)); A_mul_B!(y, A, x))
*(A::AbstractLinOp{T}, B::AbstractMatrix) where {T} =
  (C = Array{T}(size(A,1), size(B,2)); A_mul_B!(C, A, B))
*(A::AbstractMatrix, B::AbstractLinOp{T}) where {T} =
  (C = Array{T}(size(A,1), size(B,2)); A_mul_B!(C, A, B))

# scalar multiplication/division

for (f, g) in ((:(A::LinOp), :(c::Number)), (:(c::Number), :(A::LinOp)))
  @eval begin
    function *($f, $g)
      T = eltype(A)
      m, n = size(A)
      mul!  = (y, _, x) -> ( A_mul_B!(y, A, x); scale!(c, y))
      mulc! = (y, _, x) -> (Ac_mul_B!(y, A, x); scale!(c, y))
      LinOp{T}(m, n, mul!, mulc!, nothing)
    end
  end
end

for (f, g) in ((:(A::HermLinOp), :(c::Number)), (:(c::Number), :(A::HermLinOp)))
  @eval begin
    function *($f, $g)
      T = eltype(A)
      n = size(A, 1)
      mul! = (y, _, x) -> (A_mul_B!(y, A, x); scale!(c, y))
      HermLinOp{T}(n, mul!, nothing)
    end
  end
end

-(A::AbstractLinOp) = -1*A

/(A::AbstractLinOp, c::Number) = A*(1/c)
\(c::Number, A::AbstractLinOp) = (1/c)*A

# operator addition/subtraction

for (f, a) in ((:+, 1), (:-, -1))
  @eval begin
    function $f(A::AbstractLinOp{T}, B::AbstractLinOp{T}) where T
      size(A) == size(B) || throw(DimensionMismatch)
      m, n = size(A)
      alpha = T($a)
      mul!  =  gen_linop_axpy(A, B, alpha)
      mulc! = gen_linop_axpyc(A, B, alpha)
      LinOp{T}(m, n, mul!, mulc!, nothing)
    end

    function $f(A::HermLinOp{T}, B::HermLinOp{T}) where T
      size(A) == size(B) || throw(DimensionMismatch)
      n = size(A, 1)
      alpha = T($a)
      mul! = gen_linop_axpy(A, B, alpha)
      HermLinOp{T}(n, mul!, nothing)
    end
  end
end

for (f, g) in ((:axpy, :A_mul_B!), (:axpyc, :Ac_mul_B!))
  gen = Symbol("gen_linop_", f)
  fcn = Symbol("linop_", f, "!")
  @eval begin
    function $gen(A::AbstractLinOp{T}, B::AbstractLinOp{T}, alpha::T) where T
      function $fcn(
          y::StridedVecOrMat{T}, L::AbstractLinOp{T}, x::StridedVecOrMat{T}) where T
        if isnull(L._tmp) || size(get(L._tmp)) != size(y)
          L._tmp = similar(y)
        end
        tmp = get(L._tmp)
        $g( y , A, x)
        $g(tmp, B, x)
        BLAS.axpy!(alpha*one(T), tmp, y)
      end
    end
  end
end

# operator composition

function *(A::AbstractLinOp{T}, B::AbstractLinOp{T}) where T
  mA, nA = size(A)
  mB, nB = size(B)
  nA == mB || throw(DimensionMismatch)
  mul!  =  gen_linop_comp(A, B)
  mulc! = gen_linop_compc(A, B)
  LinOp{T}(mA, nB, mul!, mulc!, nothing)
end

for (f, g) in ((:comp, :A_mul_B!), (:compc, :Ac_mul_B!))
  gen = Symbol("gen_linop_", f)
  fcn = Symbol("linop_", f, "!")
  @eval begin
    function $gen(A::AbstractLinOp{T}, B::AbstractLinOp{T}) where T
      function $fcn(
          y::StridedVector{T}, L::AbstractLinOp{T}, x::StridedVector{T}) where T
        n = size(B, 1)
        if isnull(L._tmp) || length(get(L._tmp)) != n
          L._tmp = Array{T}(n)
        end
        tmp = get(L._tmp)
        $g(tmp, B,  x )
        $g( y , A, tmp)
      end
      function $fcn(
          Y::StridedMatrix{T}, L::AbstractLinOp{T}, X::StridedMatrix{T}) where T
        m = size(B, 1)
        n = size(X, 2)
        if isnull(L._tmp) || size(get(L._tmp)) != (m, n)
          L._tmp = Array{T}(m, n)
        end
        tmp = get(L._tmp)
        $g(tmp, B,  X )
        $g( Y , A, tmp)
      end
    end
  end
end
