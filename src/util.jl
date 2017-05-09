#= src/util.jl
=#

crandn{T<:Real}(::Type{T}, dims::Integer...) = convert(Array{T}, randn(dims...))
for (elty, relty) in ((:Complex64, :Float32), (:Complex128, :Float64))
  @eval begin
    crandn(::Type{$elty}, dims::Integer...) =
      reinterpret($elty, crandn($relty, 2*dims[1], dims[2:end]...), (dims...))
  end
end

function findmaxabs{T}(x::AbstractVecOrMat{T})
  m  = zero(real(T))
  mi = 0
  @inbounds for i = 1:length(x)
    t = abs(x[i])
    t < m && continue
    m  = t
    mi = i
  end
  m, mi
end

function getcols(trans::Symbol, A::AbstractMatrix, cols)
  if     trans == :n  return A[:,cols]
  elseif trans == :c  return A[cols,:]'
  end
end
function getcols(trans::Symbol, A::AbstractLinearOperator, cols)
  if     trans == :n  return A[:,cols]
  elseif trans == :c  return A'[:, cols]
  end
end

function hermitianize!(A::AbstractMatrix, uplo::Symbol=:U)
  checksquare(A)
  n = size(A, 2)
  if uplo == :U
    for j = 1:n, i = 1:j
      @inbounds A[i,j] = 0.5*(A[i,j] + conj(A[j,i]))
    end
  elseif uplo == :L
    for i = 1:n, j = 1:i
      @inbounds A[i,j] = 0.5*(A[i,j] + conj(A[j,i]))
    end
  end
  Hermitian(A, uplo)
end

function iscale!(A::AbstractMatrix, b::AbstractVector)
  m, n = size(A)
  length(b) == n || throw(DimensionMismatch)
  @inbounds for j = 1:n
    bj = b[j]
    @simd for i = 1:m
      A[i,j] /= bj
    end
  end
  A
end
function iscale!(b::AbstractVector, A::AbstractMatrix)
  m, n = size(A)
  length(b) == m || throw(DimensionMismatch)
  @inbounds for j = 1:n
    @simd for i = 1:m
      A[i,j] /= b[i]
    end
  end
  A
end

function orthcols!{T<:BlasFloat}(
    A::StridedMatrix{T}, tau::Vector{T}, work::Vector{T}; thin::Bool=true)
  m, n = size(A)
  k = min(m, n)
  A, tau, work = _LAPACK.geqrf!(A, tau, work)
  A, tau, work = _LAPACK.orgqr!(A, tau, k, work)
  if thin  A = k < n ? A[:,1:k] : A
  else     A[:,k+1:end] = 0
  end
  A, tau, work
end
orthcols!{T}(A::StridedMatrix{T}; thin::Bool=true) =
  orthcols!(A, Array{T}(1), Array{T}(1), thin=thin)[1]

function orthrows!{T<:BlasFloat}(
    A::StridedMatrix{T}, tau::Vector{T}, work::Vector{T}; thin::Bool=true)
  m, n = size(A)
  k = min(m, n)
  A, tau, work = _LAPACK.gelqf!(A, tau, work)
  A, tau, work = _LAPACK.orglq!(A, tau, k, work)
  if thin  A = k < m ? A[1:k,:] : A
  else     A[k+1:end,:] = 0
  end
  A, tau, work
end
orthrows!{T}(A::StridedMatrix{T}; thin::Bool=true) =
  orthrows!(A, Array{T}(1), Array{T}(1), thin=thin)[1]

function scalevec!(s::AbstractVector, x::AbstractVector)
  n = length(x)
  length(s) == n || throw(DimensionMismatch)
  @inbounds @simd for i = 1:n
    x[i] *= s[i]
  end
  x
end
function iscalevec!(s::AbstractVector, x::AbstractVector)
  n = length(x)
  length(s) == n || throw(DimensionMismatch)
  @inbounds @simd for i = 1:n
    x[i] /= s[i]
  end
  x
end

symrelerr(x, y) = 2*abs((x - y)/(x + y))
