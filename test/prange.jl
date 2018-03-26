#= test/prange.jl
=#

@testset "prange" begin
  n = 128
  M = matrixlib(:fourier, rand(n), rand(n))
  opts = LRAOptions(maxdet_tol=0., sketch_randn_niter=1)

  @time for (t, s) in ((:none,                 :none ),
                       (:RandomGaussian,       :randn),
                       (:RandomSubset,         :sub  ),
                       (:SRFT,                 :srft ),
                       (:SparseRandomGaussian, :sprn ))
    opts.sketch = s
    for T in (Float32, Float64, ComplexF32, ComplexF64)
      println("  $t/$T")

      rtol = 5*eps(real(T))
      approx_rtol = 100*rtol
      opts.rtol = rtol

      A = convert(Array{T}, T <: Real ? real(M) : M)
      nrm = norm(A)

      Q = prange(:n, A, opts)
      @test norm(A - Q*(Q'*A)     ) < approx_rtol*nrm
      Q = prange(:c, A, opts)
      @test norm(A -      (A*Q)*Q') < approx_rtol*nrm
      Q = prange(:b, A, opts)
      @test norm(A - Q*(Q'*A*Q)*Q') < approx_rtol*nrm
    end
  end
end
