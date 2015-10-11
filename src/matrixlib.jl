#= matrixlib.jl
=#

function matrixlib(name::Symbol, args...)
  if     name == :cauchy   return matrixlib_cauchy(args...)
  elseif name == :fourier  return matrixlib_fourier(args...)
  elseif name == :hilb     return matrixlib_hilb(args...)
  else                     throw(ArgumentError("name"))
  end
end

matrixlib_cauchy(x::AbstractVector, y::AbstractVector) = 1./broadcast(-, x, y')

matrixlib_fourier(f::AbstractVector, x::AbstractVector) = exp(-2im*pi*f*x')
matrixlib_fourier(m::Integer, n::Integer) = matrixlib_fourier(0:m-1, (0:n-1)/n)
matrixlib_fourier(n::Integer) = matrixlib_fourier(n, n)

matrixlib_hilb(m::Integer, n::Integer) = 1./(broadcast(+, 1:m, (1:n)') - 1)
matrixlib_hilb(n::Integer) = matrixlib_hilb(n, n)