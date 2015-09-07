#= src/linop.jl
=#

abstract AbstractLinearOperator{T}
typealias AbstractLinOp AbstractLinearOperator
typealias AbstractMatOrLinOp{T} Union(AbstractMatrix{T}, AbstractLinOp{T})

type LinearOperator{T} <: AbstractLinOp{T}
  m::Int
  n::Int
  mul!::Function
  mulc!::Function
end
typealias LinOp LinearOperator

type HermitianLinearOperator{T} <: AbstractLinOp{T}
  n::Int
  mul!::Function
end
typealias HermLinOp HermitianLinearOperator

function LinOp(A)
  try
    if ishermitian(A)
      return HermLinOp(A)
    end
  end
  T = eltype(A)
  m, n = size(A)
  mul!  = (y, x) ->  A_mul_B!(y, A, x)
  mulc! = (y, x) -> Ac_mul_B!(y, A, x)
  LinOp{T}(m, n, mul!, mulc!)
end

function HermLinOp(A)
  T = eltype(A)
  m, n = size(A)
  m == n || throw(DimensionMismatch)
  mul! = (y, x) ->  A_mul_B!(y, A, x)
  HermLinOp{T}(n, mul!)
end

convert{T}(::Type{Array}, A::AbstractLinOp{T}) = full(A)
convert{T}(::Type{Array{T}}, A::AbstractLinOp) = convert(Array{T}, full(A))

ctranspose{T}(A::LinOp{T}) = LinOp{T}(A.n, A.m, A.mulc!, A.mul!)
ctranspose(A::HermLinOp) = A

eltype{T}(A::AbstractLinOp{T}) = T

full(A::AbstractLinOp) = A*eye(size(A, 2))

ishermitian(A::LinOp) = false
ishermitian(A::HermLinOp) = true
issym(A::LinOp) = false
issym(A::HermLinOp) = isreal(A)

isreal{T}(A::AbstractLinOp{T}) = T <: Real

size(A::LinOp) = (A.m, A.n)
size(A::LinOp, dim::Integer) = dim == 1 ? A.m : (dim == 2 ? A.n : 1)
size(A::HermLinOp) = (A.n, A.n)
size(A::HermLinOp, dim::Integer) = (dim == 1 || dim == 2) ? A.n : 1

function transpose{T}(A::LinOp{T})
  n, m = size(A)
  mul!  = (y, x) -> (A.mulc!(y, conj(x)); conj!(y))
  mulc! = (y, x) -> ( A.mul!(y, conj(x)); conj!(y))
  LinOp{T}(m, n, mul!, mulc!)
end
function transpose{T}(A::HermLinOp{T})
  n = size(A, 1)
  mul! = (y, x) -> (A.mul!(y, conj(x)); conj!(y))
  HermLinOp{T}(n, mul!)
end

# matrix multiplication

A_mul_B!(C, A::AbstractLinOp, B::AbstractVecOrMat) = A.mul!(C, B)
Ac_mul_B!(C, A::LinOp, B::AbstractVecOrMat) = A.mulc!(C, B)
Ac_mul_B!(C, A::HermLinOp, B::AbstractVecOrMat) = A_mul_B!(C, A, B)

A_mul_B!(C, A::AbstractMatrix, B::AbstractLinOp) = ctranspose!(C, B'*A')
A_mul_Bc!(C, A::AbstractMatrix, B::AbstractLinOp) = ctranspose!(C, B*A')

*{T}(A::AbstractLinOp{T}, B::AbstractVector) =
  (C = Array(T, size(A,1)); A_mul_B!(C, A, B))
*{T}(A::AbstractLinOp{T}, B::AbstractMatrix) =
  (C = Array(T, size(A,1), size(B,2)); A_mul_B!(C, A, B))
*{T}(A::AbstractMatrix, B::AbstractLinOp{T}) =
  (C = Array(T, size(A,1), size(B,2)); A_mul_B!(C, A, B))

# scalar multiplication/division

for (f, g) in ((:(A::LinOp), :(c::Number)), (:(c::Number), :(A::LinOp)))
  @eval begin
    function *($f, $g)
      T = eltype(A)
      m, n = size(A)
      mul!  = (y, x) -> ( A_mul_B!(y, A, x); scale!(c, y))
      mulc! = (y, x) -> (Ac_mul_B!(y, A, x); scale!(c, y))
      LinOp{T}(m, n, mul!, mulc!)
    end
  end
end

for (f, g) in ((:(A::HermLinOp), :(c::Number)), (:(c::Number), :(A::HermLinOp)))
  @eval begin
    function *($f, $g)
      T = eltype(A)
      n = size(A, 1)
      mul! = (y, x) -> (A_mul_B!(y, A, x); scale!(c, y))
      HermLinOp{T}(n, mul!)
    end
  end
end

-(A::AbstractLinOp) = -1*A

/(A::AbstractLinOp, c::Number) = A*(1/c)
\(c::Number, A::AbstractLinOp) = (1/c)*A

# operator addition/subtraction
for f in (:+, :-)
  @eval begin
    function $f{T}(A::AbstractLinOp{T}, B::AbstractLinOp{T})
      size(A) == size(B) || throw(DimensionMismatch)
      m, n = size(A)
      mul!  = (y, x) -> ( A_mul_B!(y, A, x); copy!(y, $f(y, B *x)))
      mulc! = (y, x) -> (Ac_mul_B!(y, A, x); copy!(y, $f(y, B'*x)))
      LinOp{T}(m, n, mul!, mulc!)
    end

    function $f{T}(A::HermLinOp{T}, B::HermLinOp{T})
      size(A) == size(B) || throw(DimensionMismatch)
      n = size(A, 1)
      mul! = (y, x) -> (A_mul_B!(y, A, x); copy!(y, $f(y, B*x)))
      HermLinOp{T}(n, mul!)
    end
  end
end

# operator composition
function *{T}(A::AbstractLinOp{T}, B::AbstractLinOp{T})
  mA, nA = size(A)
  mB, nB = size(B)
  nA == mB || throw(DimensionMismatch)
  mul!  = (y, x) ->  A_mul_B!(y, A, B *x)
  mulc! = (y, x) -> Ac_mul_B!(y, B, A'*x)
  LinOp{T}(mA, nB, mul!, mulc!)
end