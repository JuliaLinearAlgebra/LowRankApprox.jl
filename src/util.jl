#= src/util.jl
=#

hilbert(m::Integer, n::Integer) = 1./(broadcast(+, 1:m, (1:n)') - 1)

function orthcols!{T<:BlasFloat}(A::StridedMatrix{T})
  tau = Array(T, minimum(size(A)))
  LAPACK.geqrf!(A, tau)
  LAPACK.orgqr!(A, tau)
end

function orthrows!{T<:BlasFloat}(A::StridedMatrix{T})
  tau = Array(T, minimum(size(A)))
  LAPACK.gelqf!(A, tau)
  LAPACK.orglq!(A, tau)
end

randi(a::Integer, b::Integer, dims...) =
  floor(Int, (b - a + 1)*rand(dims...)) + a

function randnt{T}(::Type{T}, dims...)
  A = randn(dims...)
  if T <: Complex
    A += im*randn(dims...)
  end
  convert(Array{T}, A)
end