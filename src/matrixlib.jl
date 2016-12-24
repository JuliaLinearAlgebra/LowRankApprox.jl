#= matrixlib.jl
=#

function matrixlib{T}(::Type{T}, name::Symbol, args...)
  if     name == :cauchy   return matrixlib_cauchy(T, args...)
  elseif name == :fourier  return matrixlib_fourier(T, args...)
  elseif name == :hilb     return matrixlib_hilb(T, args...)
  else                     throw(ArgumentError("name"))
  end
end
matrixlib(name::Symbol, args...) = matrixlib(Float64, name, args...)

function matrixlib_cauchy{T}(::Type{T}, x::AbstractVector, y::AbstractVector)
  m = length(x)
  n = length(y)
  A = Array(T, m, n)
  @inbounds for j = 1:n
    @simd for i = 1:m
      A[i,j] = 1/(x[i] - y[j])
    end
  end
  A
end

function matrixlib_fourier{S}(::Type{S}, x::AbstractVector, y::AbstractVector)
  T = eltype(complex(zero(S)))
  m = length(x)
  n = length(y)
  A = Array(T, m, n)
  @inbounds for j = 1:n, i = 1:m
    A[i,j] = exp(-2im*pi*x[i]*y[j])
  end
  A
end
matrixlib_fourier(m::Integer, n::Integer) = matrixlib_fourier(0:m-1, (0:n-1)/n)
matrixlib_fourier(n::Integer) = matrixlib_fourier(n, n)

function matrixlib_hilb{T}(::Type{T}, m::Integer, n::Integer)
  m >= 0 || throw(ArgumentError("m"))
  n >= 0 || throw(ArgumentError("n"))
  A = Array(T, m, n)
  @inbounds for j = 1:n, i = 1:m
    A[i,j] = 1/(i + j - 1)
  end
  A
end
matrixlib_hilb(n::Integer) = matrixlib_hilb(n, n)