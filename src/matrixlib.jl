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

function matrixlib_fourier{T}(::Type{T}, x::AbstractVector, y::AbstractVector)
  S = eltype(complex(zero(T)))
  m = length(x)
  n = length(y)
  A = Array(S, m, n)
  @inbounds for j = 1:n, i = 1:m
    A[i,j] = exp(-2im*pi*x[i]*y[j])
  end
  A
end
matrixlib_fourier{T}(::Type{T}, m::Integer, n::Integer) =
  matrixlib_fourier(T, 0:m-1, (0:n-1)/n)
matrixlib_fourier{T}(::Type{T}, n::Integer) = matrixlib_fourier(T, n, n)

function matrixlib_hilb{T}(::Type{T}, m::Integer, n::Integer)
  m >= 0 || throw(ArgumentError("m"))
  n >= 0 || throw(ArgumentError("n"))
  A = Array(T, m, n)
  @inbounds for j = 1:n, i = 1:m
    A[i,j] = 1/(i + j - 1)
  end
  A
end
matrixlib_hilb{T}(::Type{T}, n::Integer) = matrixlib_hilb(T, n, n)