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
    i != A.p[A.p[i]] && return false
  end
  true
end

size(A::PermMat) = (n = length(A.p); (n, n))
size(A::PermMat, dim::Integer) = (dim == 1 || dim == 2) ? length(A.p) : 1

sparse(A::RowPerm) = sparse(1:length(A.p), A.p, ones(A.p))
sparse(A::ColPerm) = sparse(A.p, 1:length(A.p), ones(A.p))

# in-place permutation routines

function rowperm!(fwd::Bool, x::StridedVector, p::Vector{Int})
  n = length(x)
  length(p) == n || throw(DimensionMismatch)
  scale!(p, -1)
  if (fwd)
    for i = 1:n
      p[i] > 0 && continue
      j    =    i
      p[j] = -p[j]
      k    =  p[j]
      while p[k] < 0
        x[j], x[k] = x[k], x[j]
        j    =    k
        p[j] = -p[j]
        k    =  p[j]
      end
    end
  else
    for i = 1:n
      p[i] > 0 && continue
      p[i] = -p[i]
      j    =  p[i]
      while p[j] < 0
        x[i], x[j] = x[j], x[i]
        p[j] = -p[j]
        j    =  p[j]
      end
    end
  end
  x
end
function rowperm!(fwd::Bool, A::StridedMatrix, p::Vector{Int})
  m, n = size(A)
  length(p) == m || throw(DimensionMismatch)
  scale!(p, -1)
  if (fwd)
    for i = 1:m
      p[i] > 0 && continue
      j    =    i
      p[j] = -p[j]
      k    =  p[j]
      while p[k] < 0
        for l = 1:n
          A[j,l], A[k,l] = A[k,l], A[j,l]
        end
        j    =    k
        p[j] = -p[j]
        k    =  p[j]
      end
    end
  else
    for i = 1:m
      p[i] > 0 && continue
      p[i] = -p[i]
      j    =  p[i]
      while p[j] < 0
        for l = 1:n
          A[i,l], A[j,l] = A[j,l], A[i,l]
        end
        p[j] = -p[j]
        j    =  p[j]
      end
    end
  end
  A
end

function colperm!(fwd::Bool, A::StridedMatrix, p::Vector{Int})
  m, n = size(A)
  length(p) == n || throw(DimensionMismatch)
  scale!(p, -1)
  if (fwd)
    for i = 1:n
      p[i] > 0 && continue
      j    =    i
      p[j] = -p[j]
      k    =  p[j]
      while p[k] < 0
        for l = 1:m
          A[l,j], A[l,k] = A[l,k], A[l,j]
        end
        j    =    k
        p[j] = -p[j]
        k    =  p[j]
      end
    end
  else
    for i = 1:n
      p[i] > 0 && continue
      p[i] = -p[i]
      j    =  p[i]
      while p[j] < 0
        for l = 1:m
          A[l,i], A[l,j] = A[l,j], A[l,i]
        end
        p[j] = -p[j]
        j    =  p[j]
      end
    end
  end
  A
end

## RowPermutation
A_mul_B!(A::RowPerm, B::StridedVecOrMat) = rowperm!(true, B, A.p)
A_mul_B!(A::StridedMatrix, B::RowPerm) = colperm!(false, A, B.p)
A_mul_Bc!(A::StridedMatrix, B::RowPerm) = colperm!(true, A, B.p)
Ac_mul_B!(A::RowPerm, B::StridedVecOrMat) = rowperm!(false, B, A.p)

## ColumnPermutation
A_mul_B!(A::ColPerm, B::StridedVecOrMat) = rowperm!(false, B, A.p)
A_mul_B!(A::StridedMatrix, B::ColPerm) = colperm!(true, A, B.p)
A_mul_Bc!(A::StridedMatrix, B::ColPerm) = colperm!(false, A, B.p)
Ac_mul_B!(A::ColPerm, B::StridedVecOrMat) = rowperm!(true, B, A.p)

## transpose multiplication
A_mul_Bt!(A::StridedMatrix, B::PermMat) = A_mul_Bc!(A, B)
At_mul_B!(A::PermMat, B::StridedVecOrMat) = Ac_mul_B!(A, B)

# standard operations

## left-multiplication
for (f, f!) in ((:*,        :A_mul_B!),
                (:Ac_mul_B, :Ac_mul_B!),
                (:At_mul_B, :At_mul_B!))
  for t in (:RowPerm, :ColPerm)
    @eval $f(A::$t, B::StridedVecOrMat) = $f!(A, copy(B))
  end
end

## right-multiplication
for (f, f!) in ((:*,        :A_mul_B!),
                (:A_mul_Bc, :A_mul_Bc!),
                (:A_mul_Bt, :A_mul_Bt!))
  for t in (:RowPerm, :ColPerm)
    @eval $f(A::StridedMatrix, B::$t) = $f!(copy(A), B)
  end
end

## operations on matrix copies
A_mul_Bc(A::PermMat, B::StridedMatrix) = A_mul_B!(A, B')
A_mul_Bt(A::PermMat, B::StridedMatrix) = A_mul_B!(A, B.')
Ac_mul_B(A::StridedMatrix, B::PermMat) = A_mul_B!(A', B)
Ac_mul_Bc(A::PermMat, B::StridedMatrix) = Ac_mul_B!(A, B')
Ac_mul_Bc(A::StridedMatrix, B::PermMat) = A_mul_Bc!(A', B)
At_mul_B(A::StridedMatrix, B::PermMat) = A_mul_B!(A.', B)
At_mul_Bt(A::PermMat, B::StridedMatrix) = At_mul_B!(A, B.')
At_mul_Bt(A::StridedMatrix, B::PermMat) = A_mul_Bt!(A.', B)