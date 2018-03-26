#= test/permute.jl
=#

@testset "permute" begin
  n = 10
  p = randperm(n)

  @time for t in (RowPermutation, ColumnPermutation)
    println("  $t")

    A = t(p)
    P = full(A)
    @test P == sparse(A)
    @test P == A*eye(n)
    @test P == eye(n)*A

    x = rand(n)
    @test A  *x == P  *x
    @test A' *x == P' *x
    @test A.'*x == P.'*x

    B = rand(n, n)
    @test A  *B   == P  *B
    @test A  *B'  == P  *B'
    @test A  *B.' == P  *B.'
    @test A' *B   == P' *B
    @test A' *B'  == P' *B'
    @test A.'*B   == P.'*B
    @test A.'*B.' == P.'*B.'
    @test B  *A   == B  *P
    @test B  *A'  == B  *P'
    @test B  *A.' == B  *P.'
    @test B' *A   == B' *P
    @test B' *A'  == B' *P'
    @test B.'*A   == B.'*P
    @test B.'*A.' == B.'*P.'
  end
end
