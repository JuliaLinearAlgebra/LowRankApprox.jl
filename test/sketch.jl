#= test/sketch.jl
=#

println("sketch.jl")
tic()

m = 20
n = 10
A = rand(m, n)
rank = 5
opts = LRAOptions(sketch_randn_niter=1)

for (t, s) in ((RandomGaussian, :randn), (RandomSubset, :subs), (SRFT, :srft))
  opts.sketch = s
  for T in (Float64, Complex128)
    println("  $t/$T")

    S = sketch(:left, :n, A, rank, opts)
    @test size(S) == (rank, n)
    S = sketch(:left, :c, A, rank, opts)
    @test size(S) == (rank, m)
    S = sketch(:right, :n, A, rank, opts)
    @test size(S) == (m, rank)
    S = sketch(:right, :c, A, rank, opts)
    @test size(S) == (n, rank)
  end
end

toc()