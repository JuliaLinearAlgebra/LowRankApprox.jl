#= src/util.jl
=#

crandn{T<:Real}(::Type{T}, dims::Integer...) = convert(Array{T}, randn(dims...))
for (elty, relty) in ((:Complex64, :Float32), (:Complex128, :Float64))
  @eval begin
    crandn(::Type{$elty}, dims::Integer...) =
      reinterpret($elty, crandn($relty, 2*dims[1], dims[2:end]...), (dims...))
  end
end

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

function scalevec!(s::StridedVector, x::StridedVector)
  for i = 1:length(x)
    x[i] *= s[i]
  end
  x
end