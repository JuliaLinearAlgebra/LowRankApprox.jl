#= test/snorm.jl
=#

println("snorm.jl")
tic()

m = 20
n = 10

for T in (Float32, Float64, Complex64, Complex128)
  println("  $T")

  rtol = 100*eps(real(T))

  A = rand(T, m, n)
  nrm = norm(A)
  @test_approx_eq_eps nrm snorm(A) rtol*nrm

  A = rand(T, n, n)
  A += A'
  nrm = norm(A)
  @test_approx_eq_eps nrm snorm(A) rtol*nrm

  B = rand(T, size(A))
  nrm = norm(A - B)
  @test_approx_eq_eps nrm snormdiff(A, B) rtol*nrm
end

toc()