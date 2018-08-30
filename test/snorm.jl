#= test/snorm.jl
=#

@testset "snorm" begin
  m = 20
  n = 10

  @time for T in (Float32, Float64, ComplexF32, ComplexF64)
    let A
      println("  $T")

      rtol = 100*eps(real(T))

      A = rand(T, m, n)
      nrm = opnorm(A)
      @test ≈(nrm, snorm(A); atol = rtol*nrm)

      A = rand(T, n, n)
      A += A'
      nrm = opnorm(A)
      @test ≈(nrm, snorm(A); atol = rtol*nrm)

      B = rand(T, size(A))
      nrm = opnorm(A - B)
      @test ≈(nrm, snormdiff(A, B); atol = rtol*nrm)
    end
  end
end
