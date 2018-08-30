#= test/linop.jl
=#

@testset "linop" begin
  n = 10

  @time for (t, herm) in ((LinearOperator, false), (HermitianLinearOperator, true))
    for T in (Float32, Float64, ComplexF32, ComplexF64)
      println("  $t/$T")

      A = rand(T, n, n)
      if herm
        A += A'
      end

      L = LinearOperator(A)
      @test isa(L, t)

      F = Matrix(L)

      @test A ≈ F
      @test F' ≈ Matrix(L')
      @test A ≈ L*Matrix(1.0I,n,n)
      @test A ≈ Matrix(1.0I,n,n)*L

      x = rand(T, n)
      @test A  *x ≈ L  *x
      @test A' *x ≈ L' *x
      @test transpose(A)*x ≈ transpose(L)*x

      B = rand(T, n, n)
      @test A  *B   ≈ L  *B
      @test A  *B'  ≈ L  *B'
      @test A  *transpose(B) ≈ L  *transpose(B)
      @test A' *B   ≈ L' *B
      @test A' *B'  ≈ L' *B'
      @test transpose(A)*B   ≈ transpose(L)*B
      @test transpose(A)*transpose(B) ≈ transpose(L)*transpose(B)
      @test B  *A   ≈ B  *L
      @test B  *A'  ≈ B  *L'
      @test B  *transpose(A) ≈ B  *transpose(L)
      @test B' *A   ≈ B' *L
      @test B' *A'  ≈ B' *L'
      @test transpose(B)*A   ≈ transpose(B)*L
      @test transpose(B)*transpose(A) ≈ transpose(B)*transpose(L)

      c = rand()
      @test (c*A) *x ≈ (c*L) *x
      @test (c*A)'*x ≈ (c*L)'*x
      @test (c\A) *x ≈ (c\L) *x
      @test (c\A)'*x ≈ (c\L)'*x
      @test (A/c) *x ≈ (L/c) *x
      @test (A/c)'*x ≈ (L/c)'*x

      M = c*L
      M = L + M
      @test isa(M, t)
      @test (1 + c)*(A*x) ≈ M*x

      M = c*L
      M = L - M
      @test isa(M, t)
      @test (1 - c)*(A*x) ≈ M*x

      M = L^2
      @test A*(A*x) ≈ M*x

      M = convert(LinearOperator{T}, L)
      @test A*x == M*x
      M = convert(LinearOperator, L)
      @test A*x == M*x
    end
  end
end
