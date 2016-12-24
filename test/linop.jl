#= test/linop.jl
=#

println("linop.jl")
tic()

n = 10

for (t, herm) in ((:LinearOperator, false), (:HermitianLinearOperator, true))
  for T in (Float32, Float64, Complex64, Complex128)
    println("  $t/$T")

    A = rand(T, n, n)
    if herm
      A += A'
    end

    L = LinearOperator(A)
    @eval @test isa($L, $t)

    F = full(L)
    @test_approx_eq A F
    @test_approx_eq A L*eye(n)
    @test_approx_eq A eye(n)*L

    x = rand(T, n)
    @test_approx_eq A  *x L  *x
    @test_approx_eq A' *x L' *x
    @test_approx_eq A.'*x L.'*x

    B = rand(T, n, n)
    @test_approx_eq A  *B   L  *B
    @test_approx_eq A  *B'  L  *B'
    @test_approx_eq A  *B.' L  *B.'
    @test_approx_eq A' *B   L' *B
    @test_approx_eq A' *B'  L' *B'
    @test_approx_eq A.'*B   L.'*B
    @test_approx_eq A.'*B.' L.'*B.'
    @test_approx_eq B  *A   B  *L
    @test_approx_eq B  *A'  B  *L'
    @test_approx_eq B  *A.' B  *L.'
    @test_approx_eq B' *A   B' *L
    @test_approx_eq B' *A'  B' *L'
    @test_approx_eq B.'*A   B.'*L
    @test_approx_eq B.'*A.' B.'*L.'

    c = rand()
    @test_approx_eq (c*A) *x (c*L) *x
    @test_approx_eq (c*A)'*x (c*L)'*x
    @test_approx_eq (c\A) *x (c\L) *x
    @test_approx_eq (c\A)'*x (c\L)'*x
    @test_approx_eq (A/c) *x (L/c) *x
    @test_approx_eq (A/c)'*x (L/c)'*x

    M = c*L
    M = L + M
    @eval @test isa($M, $t)
    @test_approx_eq (1 + c)*(A*x) M*x

    M = c*L
    M = L - M
    @eval @test isa($M, $t)
    @test_approx_eq (1 - c)*(A*x) M*x

    M = L^2
    @test_approx_eq A*(A*x) M*x
  end
end

toc()