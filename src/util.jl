#= src/util.jl
=#

# redefinition of non-specific methods in Base and associated routines

_randn{T<:Real}(::Type{T}, dims::Integer...) = convert(Array{T}, randn(dims...))
for (elty, relty) in ((:Complex64, :Float32), (:Complex128, :Float64))
  @eval begin
    _randn(::Type{$elty}, dims::Integer...) =
      reinterpret($elty, _randn($relty, 2*dims[1], dims[2:end]...), (dims...))
  end
end

function _scale!(s::AbstractVector, x::AbstractVector)
  n = length(x)
  length(s) == n || throw(DimensionMismatch)
  for i = 1:n
    x[i] *= s[i]
  end
  x
end
_scale!(A::AbstractMatrix, b::AbstractVector) = scale!(A, b)
_scale!(b::AbstractVector, A::AbstractMatrix) = scale!(b, A)

function _iscale!(s::AbstractVector, x::AbstractVector)
  n = length(x)
  length(s) == n || throw(DimensionMismatch)
  for i = 1:n
    x[i] /= s[i]
  end
  x
end
function _iscale!(A::AbstractMatrix, b::AbstractVector)
  m, n = size(A)
  length(b) == n || throw(DimensionMismatch)
  for j = 1:n
    bj = b[j]
    for i = 1:m
      A[i,j] /= bj
    end
  end
  A
end
function _iscale!(b::AbstractVector, A::AbstractMatrix)
  m, n = size(A)
  length(b) == m || throw(DimensionMismatch)
  for j = 1:n, i = 1:m
    A[i,j] /= b[i]
  end
  A
end

# other utility routines

hilbert(m::Integer, n::Integer) = 1./(broadcast(+, 1:m, (1:n)') - 1)

function orthcols!{T<:BlasFloat}(A::StridedMatrix{T}; thin::Bool=true)
  k = minimum(size(A))
  tau = Array(T, k)
  LAPACK.geqrf!(A, tau)
  Q = LAPACK.orgqr!(A, tau)
  if thin  return Q
  else     (A[:,k+1:end] = 0; return A)
  end
end

function orthrows!{T<:BlasFloat}(A::StridedMatrix{T}; thin::Bool=true)
  k = minimum(size(A))
  tau = Array(T, k)
  LAPACK.gelqf!(A, tau)
  Q = LAPACK.orglq!(A, tau)
  if thin  return Q
  else     (A[k+1:end,:] = 0; return A)
  end
end