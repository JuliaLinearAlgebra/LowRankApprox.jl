#= test/sketch.jl
=#

@testset "sketch" begin
  m = 20
  n = 10
  M = rand(ComplexF64, m, n)
  r = 5
  opts = LRAOptions(sketch_randn_niter=1)

  @time for (t, s) in ((:RandomGaussian,       :randn),
                       (:RandomSubset,         :sub  ),
                       (:SRFT,                 :srft ),
                       (:SparseRandomGaussian, :sprn ))
    opts.sketch = s
    for T in (Float32, Float64, ComplexF32, ComplexF64)
      let A
          println("  $t/$T")

          A = convert(Array{T}, T <: Real ? real(M) : M)

          S = sketch(:left,  :n, A, r, opts)
          @test size(S) == (r, n)
          S = sketch(:left,  :c, A, r, opts)
          @test size(S) == (r, m)
          S = sketch(:right, :n, A, r, opts)
          @test size(S) == (m, r)
          S = sketch(:right, :c, A, r, opts)
          @test size(S) == (n, r)
      end
    end
  end
end
