#= src/permute.jl
=#

abstract PermutationMatrix <: AbstractMatrix{Int}
typealias PermMat PermutationMatrix

type RowPermutation <: PermMat
  p::Vector{Int}
end
typealias RowPerm RowPermutation

type ColumnPermutation <: PermMat
  p::Vector{Int}
end
typealias ColPerm ColumnPermutation

convert(::Type{Array}, A::PermMat) = full(A)
convert{T}(::Type{Array{T}}, A::PermMat) = convert(Array{T}, full(A))

copy(A::RowPerm) = RowPerm(copy(A.p))
copy(A::ColPerm) = ColPerm(copy(A.p))

ctranspose(A::PermMat) = transpose(A)
transpose(A::RowPerm) = ColPerm(A.p)
transpose(A::ColPerm) = RowPerm(A.p)

function full(A::RowPerm)
  n = length(A.p)
  P = zeros(Int, n, n)
  for i = 1:n
    j = A.p[i]
    P[i,j] = 1
  end
  P
end
function full(A::ColPerm)
  n = length(A.p)
  P = zeros(Int, n, n)
  for j = 1:n
    i = A.p[j]
    P[i,j] = 1
  end
  P
end

getindex(A::RowPerm, i::Integer, j::Integer) = A.p[i] == j ? 1 : 0
getindex(A::ColPerm, i::Integer, j::Integer) = A.p[j] == i ? 1 : 0

ishermitian(A::PermMat) = issym(A)
function issym(A::PermMat)
  for i = 1:length(A.p)
    j = A.p[i]
    if i != A.p[j]
      return false
    end
  end
  true
end

size(A::PermMat) = (n = length(A.p); (n, n))
size(A::PermMat, dim::Integer) = (dim == 1 || dim == 2) ? length(A.p) : 1

sparse(A::RowPerm) = sparse(1:length(A.p), A.p, ones(A.p))
sparse(A::ColPerm) = sparse(A.p, 1:length(A.p), ones(A.p))

# BLAS/LAPACK routines (in-place)

## RowPermutation
A_mul_B!{T<:BlasFloat}(A::RowPerm, B::StridedVecOrMat{T}) =
  _LAPACK.lapmr!(1, B, A.p)
A_mul_B!{T<:BlasFloat}(A::StridedMatrix{T}, B::RowPerm) =
  _LAPACK.lapmt!(0, A, B.p)
A_mul_Bc!{T<:BlasFloat}(A::StridedMatrix{T}, B::RowPerm) =
  _LAPACK.lapmt!(1, A, B.p)
Ac_mul_B!{T<:BlasFloat}(A::RowPerm, B::StridedVecOrMat{T}) =
  _LAPACK.lapmr!(0, B, A.p)

## ColumnPermutation
A_mul_B!{T<:BlasFloat}(A::ColPerm, B::StridedVecOrMat{T}) =
  _LAPACK.lapmr!(0, B, A.p)
A_mul_B!{T<:BlasFloat}(A::StridedMatrix{T}, B::ColPerm) =
  _LAPACK.lapmt!(1, A, B.p)
A_mul_Bc!{T<:BlasFloat}(A::StridedMatrix{T}, B::ColPerm) =
  _LAPACK.lapmt!(0, A, B.p)
Ac_mul_B!{T<:BlasFloat}(A::ColPerm, B::StridedVecOrMat{T}) =
  _LAPACK.lapmr!(1, B, A.p)

## transpose multiplication
A_mul_Bt!{T}(A::StridedMatrix{T}, B::PermMat) = A_mul_Bc!(A, B)
At_mul_B!{T}(A::PermMat, B::StridedVecOrMat{T}) = Ac_mul_B!(A, B)

# standard operations

## left-multiplication
for (f, f!) in ((:*,        :A_mul_B!),
                (:Ac_mul_B, :Ac_mul_B!),
                (:At_mul_B, :At_mul_B!))
  for t in (:RowPerm, :ColPerm)
    @eval $f{T}(A::$t, B::StridedVecOrMat{T}) = $f!(A, copy(B))
  end
end

## right-multiplication
for (f, f!) in ((:*,        :A_mul_B!),
                (:A_mul_Bc, :A_mul_Bc!),
                (:A_mul_Bt, :A_mul_Bt!))
  for t in (:RowPerm, :ColPerm)
    @eval $f{T}(A::StridedMatrix{T}, B::$t) = $f!(copy(A), B)
  end
end

## operations on matrix copies
A_mul_Bc{T}(A::PermMat, B::StridedMatrix{T}) = A_mul_B!(A, B')
A_mul_Bt{T}(A::PermMat, B::StridedMatrix{T}) = A_mul_B!(A, B.')
Ac_mul_B{T}(A::StridedMatrix{T}, B::PermMat) = A_mul_B!(A', B)
Ac_mul_Bc{T}(A::PermMat, B::StridedMatrix{T}) = Ac_mul_B!(A, B')
Ac_mul_Bc{T}(A::StridedMatrix{T}, B::PermMat) = A_mul_Bc!(A', B)
At_mul_B{T}(A::StridedMatrix{T}, B::PermMat) = A_mul_B!(A.', B)
At_mul_Bt{T}(A::PermMat, B::StridedMatrix{T}) = At_mul_B!(A, B.')
At_mul_Bt{T}(A::StridedMatrix{T}, B::PermMat) = A_mul_Bt!(A.', B)