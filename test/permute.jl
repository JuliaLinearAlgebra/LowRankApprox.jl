#= test/permute.jl
=#

println("permute.jl")
tic()

n = 10
p = randperm(n)

for t in (:RowPermutation, :ColumnPermutation)
  println("  $t")

  @eval A = $t(p)
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

toc()