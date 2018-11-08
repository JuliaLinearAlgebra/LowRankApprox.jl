#= src/permute.jl
=#

abstract type PermutationMatrix <: AbstractMatrix{Int} end
const PermMat = PermutationMatrix

mutable struct RowPermutation <: PermMat
  p::Vector{Int}
end
const RowPerm = RowPermutation

mutable struct ColumnPermutation <: PermMat
  p::Vector{Int}
end
const ColPerm = ColumnPermutation

convert(::Type{Array}, A::PermMat) = Matrix(A)
convert(::Type{Array{T}}, A::PermMat) where {T} = convert(Array{T}, Matrix(A))

copy(A::RowPerm) = RowPerm(copy(A.p))
copy(A::ColPerm) = ColPerm(copy(A.p))

adjoint(A::PermMat) = transpose(A)
transpose(A::RowPerm) = ColPerm(A.p)
transpose(A::ColPerm) = RowPerm(A.p)

function Matrix(A::RowPerm)
  n = length(A.p)
  P = zeros(Int, n, n)
  for i = 1:n
    j = A.p[i]
    P[i,j] = 1
  end
  P
end
function Matrix(A::ColPerm)
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
function issymmetric(A::PermMat)
  for i = 1:length(A.p)
    i != A.p[A.p[i]] && return false
  end
  true
end

size(A::PermMat) = (n = length(A.p); (n, n))
size(A::PermMat, dim::Integer) = (dim == 1 || dim == 2) ? length(A.p) : 1

sparse(A::RowPerm) = sparse(1:length(A.p), A.p, fill(1.0,length(A.p)))
sparse(A::ColPerm) = sparse(A.p, 1:length(A.p), fill(1.0,length(A.p)))

# in-place permutation routines

function rowperm!(fwd::Bool, x::AbstractVector, p::Vector{Int})
  n = length(x)
  length(p) == n || throw(DimensionMismatch)
  rmul!(p, -1)
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
function rowperm!(fwd::Bool, A::AbstractMatrix, p::Vector{Int})
  m, n = size(A)
  length(p) == m || throw(DimensionMismatch)
  rmul!(p, -1)
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

function colperm!(fwd::Bool, A::AbstractMatrix, p::Vector{Int})
  m, n = size(A)
  length(p) == n || throw(DimensionMismatch)
  rmul!(p, -1)
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
  lmul!(A::RowPerm, B::AbstractVecOrMat) = rowperm!(true, B, A.p)
  rmul!(A::AbstractMatrix, B::RowPerm) = colperm!(false, A, B.p)

  rmul!(A::AbstractMatrix, Bc::Adjoint{<:Any,<:RowPerm}) = colperm!(true, A, parent(Bc).p)
  lmul!(Ac::Adjoint{<:Any,<:RowPerm}, B::AbstractVecOrMat) = rowperm!(false, B, parent(Ac).p)

  ## ColumnPermutation
  lmul!(A::ColPerm, B::AbstractVecOrMat) = rowperm!(false, B, A.p)
  rmul!(A::AbstractMatrix, B::ColPerm) = colperm!(true, A, B.p)

  rmul!(A::AbstractMatrix, Bc::Adjoint{<:Any,<:ColPerm}) = colperm!(false, A, parent(Bc).p)
  lmul!(Ac::Adjoint{<:Any,<:ColPerm}, B::AbstractVecOrMat) = rowperm!(true, B, parent(Ac).p)

  ## transpose multiplication
  rmul!(A::AbstractMatrix, Bt::Transpose{<:Any,<:PermMat}) = rmul!(A, parent(Bt)')
  lmul!(At::Transpose{<:Any,<:PermMat}, B::AbstractVecOrMat) = lmul!(parent(At)', B)




# standard operations

  for t in (:RowPerm, :ColPerm)
    @eval begin
      ## left-multiplication
      *(A::$t, B::AbstractVector) = lmul!(A, copy(B))
      *(A::$t, B::AbstractMatrix) = lmul!(A, copy(B))
      *(A::Adjoint{<:Any,<:$t}, B::AbstractVector) = lmul!(A, copy(B))
      *(A::Adjoint{<:Any,<:$t}, B::AbstractMatrix) = lmul!(A, copy(B))
      *(A::Transpose{<:Any,<:$t}, B::AbstractVector) = lmul!(A, copy(B))
      *(A::Transpose{<:Any,<:$t}, B::AbstractMatrix) = lmul!(A, copy(B))

      ## right-multiplication
      *(A::AbstractMatrix, B::$t) = rmul!(copy(A), B)
      *(A::AbstractMatrix, B::Adjoint{<:Any,<:$t}) = rmul!(copy(A), B)
      *(A::AbstractMatrix, B::Transpose{<:Any,<:$t}) = rmul!(copy(A), B)
    end
  end
