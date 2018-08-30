#= test/permute.jl
=#

@testset "permute" begin
  n = 10
  p = randperm(n)

  @time for t in (RowPermutation, ColumnPermutation)
    println("  $t")

    A = t(p)
    P = Matrix(A)
    @test P == sparse(A)
    @test P == A*Matrix(I,n,n)
    @test P == Matrix(I,n,n)*A

    x = rand(n)
    @test A  *x == P  *x
    @test A' *x == P' *x
    @test transpose(A)*x == transpose(P)*x

    B = rand(n, n)
    @test A  *B   == P  *B
    @test A  *B'  == P  *B'
    @test A  *transpose(B) == P  *transpose(B)
    @test A' *B   == P' *B
    @test A' *B'  == P' *B'
    @test transpose(A)*B   == transpose(P)*B
    @test transpose(A)*transpose(B) == transpose(P)*transpose(B)
    @test B  *A   == B  *P
    @test B  *A'  == B  *P'
    @test B  *transpose(A) == B  *transpose(P)
    @test B' *A   == B' *P
    @test B' *A'  == B' *P'
    @test transpose(B)*A   == transpose(B)*P
    @test transpose(B)*transpose(A) == transpose(B)*transpose(P)
  end
end
