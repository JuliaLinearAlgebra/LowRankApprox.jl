#= test/snorm.jl
=#

println("snorm.jl")
tic()

m = 20
n = 10
rtol = 1e-6

for T in (Float64, Complex128)
  println("  $T")

  A = rand(T, m, n)
  nrm = norm(A)
  @test_approx_eq_eps nrm snorm(A, rtol) rtol*nrm

  A = rand(T, n, n)
  A += A'
  nrm = norm(A)
  @test_approx_eq_eps nrm snorm(A, rtol) rtol*nrm

  B = rand(T, size(A))
  nrm = norm(A - B)
  @test_approx_eq_eps nrm snormdiff(A, B, rtol) rtol*nrm
end

toc()