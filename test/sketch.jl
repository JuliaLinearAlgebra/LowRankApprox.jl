#= test/sketch.jl
=#

println("sketch.jl")
tic()

m = 20
n = 10
M = rand(Complex128, m, n)
rank = 5
opts = LRAOptions(sketch_randn_niter=1)

for (t, s) in ((:RandomGaussian,       :randn),
               (:RandomSubset,         :sub  ),
               (:SRFT,                 :srft ),
               (:SparseRandomGaussian, :sprn ))
  opts.sketch = s
  for T in (Float32, Float64, Complex64, Complex128)
    println("  $t/$T")

    A = convert(Array{T}, T <: Real ? real(M) : M)

    S = sketch(:left,  :n, A, rank, opts)
    @test size(S) == (rank, n)
    S = sketch(:left,  :c, A, rank, opts)
    @test size(S) == (rank, m)
    S = sketch(:right, :n, A, rank, opts)
    @test size(S) == (m, rank)
    S = sketch(:right, :c, A, rank, opts)
    @test size(S) == (n, rank)
  end
end

toc()