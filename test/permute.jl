#= test/permute.jl
=#

println("permute.jl")
tic()

n = 10
p = randperm(n)

for t in (:RowPermutation, :ColumnPermutation)
  @eval A = $t(p)
  P = full(A)
  @test P == sparse(A)
  @test P == A*eye(n)
  @test P == eye(n)*A

  for T in (Float32, Float64, Complex64, Complex128)
    println("  $t/$T")

    x = rand(T, n)
    @test A  *x == P  *x
    @test A' *x == P' *x
    @test A.'*x == P.'*x

    B = rand(T, n, n)
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

toc()