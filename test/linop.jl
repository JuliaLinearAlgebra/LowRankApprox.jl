#= test/linop.jl
=#

println("linop.jl")

n = 10

@time for (t, herm) in ((:LinearOperator, false), (:HermitianLinearOperator, true))
  for T in (Float32, Float64, Complex64, Complex128)
    println("  $t/$T")

    A = rand(T, n, n)
    if herm
      A += A'
    end

    L = LinearOperator(A)
    @eval @test isa($L, $t)

    F = full(L)

    @test A ≈ F
    @test F' ≈ full(L')
    @test A ≈ L*eye(n)
    @test A ≈ eye(n)*L

    x = rand(T, n)
    @test A  *x ≈ L  *x
    @test A' *x ≈ L' *x
    @test A.'*x ≈ L.'*x

    B = rand(T, n, n)
    @test A  *B   ≈ L  *B
    @test A  *B'  ≈ L  *B'
    @test A  *B.' ≈ L  *B.'
    @test A' *B   ≈ L' *B
    @test A' *B'  ≈ L' *B'
    @test A.'*B   ≈ L.'*B
    @test A.'*B.' ≈ L.'*B.'
    @test B  *A   ≈ B  *L
    @test B  *A'  ≈ B  *L'
    @test B  *A.' ≈ B  *L.'
    @test B' *A   ≈ B' *L
    @test B' *A'  ≈ B' *L'
    @test B.'*A   ≈ B.'*L
    @test B.'*A.' ≈ B.'*L.'

    c = rand()
    @test (c*A) *x ≈ (c*L) *x
    @test (c*A)'*x ≈ (c*L)'*x
    @test (c\A) *x ≈ (c\L) *x
    @test (c\A)'*x ≈ (c\L)'*x
    @test (A/c) *x ≈ (L/c) *x
    @test (A/c)'*x ≈ (L/c)'*x

    M = c*L
    M = L + M
    @eval @test isa($M, $t)
    @test (1 + c)*(A*x) ≈ M*x

    M = c*L
    M = L - M
    @eval @test isa($M, $t)
    @test (1 - c)*(A*x) ≈ M*x

    M = L^2
    @test A*(A*x) ≈ M*x
  end
end
