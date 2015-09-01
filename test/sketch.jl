#= test/sketch.jl
=#

println("sketch.jl")
tic()

m = 20
n = 10
A = rand(Complex128, m, n)
rank = 5
opts = LRAOptions(sketch_randn_niter=1)

for (t, s) in ((:RandomGaussian,       :randn),
               (:RandomSubset,         :subs),
               (:SRFT,                 :srft),
               (:SparseRandomGaussian, :sprn))
  opts.sketch = s
  for T in (Float32, Float64, Complex64, Complex128)
    println("  $t/$T")

    B = T <: Real ? real(A) : A
    B = convert(Array{T}, B)

    S = sketch(:left, :n, B, rank, opts)
    @test size(S) == (rank, n)
    S = sketch(:left, :c, B, rank, opts)
    @test size(S) == (rank, m)
    S = sketch(:right, :n, B, rank, opts)
    @test size(S) == (m, rank)
    S = sketch(:right, :c, B, rank, opts)
    @test size(S) == (n, rank)
  end
end

toc()